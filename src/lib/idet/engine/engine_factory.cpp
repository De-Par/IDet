/**
 * @file engine_factory.cpp
 * @ingroup idet_engine
 * @brief Engine factory implementation (registry-based dispatch).
 *
 * @details
 * Implements @ref idet::engine::create_engine via a small static registry of engine
 * descriptors instead of a hard-coded switch on @ref idet::EngineKind. Each descriptor
 * captures everything the factory needs to know about a backend:
 *
 * - The @ref idet::EngineKind it handles.
 * - The @ref idet::Task it operates on (so we can validate the @c (task, engine) pair).
 * - A `validate` hook to enforce engine-specific configuration constraints
 *   (e.g. DBNet's @c bin_thresh / @c box_thresh / @c unclip ranges).
 * - A `ctor` hook that allocates and constructs the concrete engine.
 *
 * Adding a new family is therefore one entry in @ref engine_registry() plus the
 * corresponding @c IEngine subclass. No changes to this file's dispatch logic, no new
 * @c switch arms, no plumbing in @c idet.cpp's validate() or setup().
 *
 * Exception safety:
 * - The factory is @c noexcept; constructor exceptions are caught and translated into
 *   @ref idet::Status so they never cross the C++ ABI boundary.
 */

#include "engine/engine_factory.h"

#include "engine/dbnet.h"
#include "engine/scrfd.h"
#include "engine/yolo.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace idet::engine {

namespace {

/**
 * @brief Engine-specific validator hook.
 *
 * @details
 * Allows the registry to centralize engine-specific configuration checks. The default
 * generic validation (@ref idet::DetectorConfig::validate) is invoked separately and
 * remains in idet.cpp for compatibility with the public API.
 */
using ValidateFn = Status (*)(const DetectorConfig& cfg) noexcept;

/**
 * @brief Engine constructor hook.
 *
 * @details
 * Wraps construction of a concrete engine to keep the registry POD-friendly. Each entry
 * is a free function that catches exceptions inside; the factory still wraps the call
 * in try/catch to satisfy the noexcept contract.
 */
using ConstructFn = Result<std::unique_ptr<IEngine>> (*)(const DetectorConfig& cfg);

/**
 * @brief Single registry entry describing an engine family.
 */
struct EngineDescriptor {
    EngineKind kind;       ///< Concrete engine identifier.
    Task task;             ///< Task this engine serves.
    const char* name;      ///< Human-readable name (used in error messages).
    ValidateFn validate;   ///< Optional engine-specific validator (may be nullptr).
    ConstructFn construct; ///< Constructor wrapper.
};

/** @brief DBNet-specific configuration constraints. */
static Status validate_dbnet_(const DetectorConfig& cfg) noexcept {
    if (!(cfg.infer.bin_thresh > 0.f && cfg.infer.bin_thresh < 1.f))
        return Status::Invalid("DBNet: bin_thresh must be in (0,1)");
    if (!(cfg.infer.box_thresh > 0.f && cfg.infer.box_thresh < 1.f))
        return Status::Invalid("DBNet: box_thresh must be in (0,1)");
    if (!(cfg.infer.unclip > 0.f)) return Status::Invalid("DBNet: unclip must be > 0");
    return Status::Ok();
}

/** @brief SCRFD-specific configuration constraints. */
static Status validate_scrfd_(const DetectorConfig& cfg) noexcept {
    if (!(cfg.infer.box_thresh > 0.f && cfg.infer.box_thresh < 1.f))
        return Status::Invalid("SCRFD: box_thresh must be in (0,1)");
    return Status::Ok();
}

/** @brief YOLO-specific configuration constraints. */
static Status validate_yolo_(const DetectorConfig& cfg) noexcept {
    if (!(cfg.infer.box_thresh > 0.f && cfg.infer.box_thresh < 1.f))
        return Status::Invalid("YOLO: box_thresh must be in (0,1)");
    return Status::Ok();
}

template <typename T>
static Result<std::unique_ptr<IEngine>> make_(const DetectorConfig& cfg) {
    std::unique_ptr<IEngine> p = std::make_unique<T>(cfg);
    return Result<std::unique_ptr<IEngine>>::Ok(std::move(p));
}

/**
 * @brief Static registry of supported engine descriptors.
 *
 * @details
 * The registry is the single source of truth for which engine kinds and tasks are
 * supported. A function-local @c static array gives us POD initialization without
 * dynamic-init order issues.
 */
const std::array<EngineDescriptor, 3>& engine_registry() noexcept {
    static const std::array<EngineDescriptor, 3> table = {
        EngineDescriptor{EngineKind::DBNet, Task::Text, "DBNet", &validate_dbnet_, &make_<DBNet>},
        EngineDescriptor{EngineKind::SCRFD, Task::Face, "SCRFD", &validate_scrfd_, &make_<SCRFD>},
        EngineDescriptor{EngineKind::Yolo, Task::Cloth, "YOLO", &validate_yolo_, &make_<YOLO>},
    };
    return table;
}

/** @brief Lookup an engine descriptor by kind. */
static const EngineDescriptor* find_descriptor_(EngineKind kind) noexcept {
    const auto& table = engine_registry();
    auto it = std::find_if(table.begin(), table.end(),
                           [&](const EngineDescriptor& d) noexcept { return d.kind == kind; });
    return (it == table.end()) ? nullptr : &(*it);
}

} // namespace

Status engine_validate_specific(const DetectorConfig& cfg) noexcept {
    const auto* desc = find_descriptor_(cfg.engine);
    if (!desc) return Status::Unsupported("engine_factory: unsupported EngineKind");
    if (desc->task != cfg.task) return Status::Invalid("engine_factory: engine/task mismatch");
    return desc->validate ? desc->validate(cfg) : Status::Ok();
}

EngineKind engine_default_for_task(Task task) noexcept {
    const auto& table = engine_registry();
    auto it = std::find_if(table.begin(), table.end(),
                           [&](const EngineDescriptor& d) noexcept { return d.task == task; });
    return (it == table.end()) ? EngineKind::None : it->kind;
}

Result<std::unique_ptr<IEngine>> create_engine(const DetectorConfig& cfg) noexcept {
    {
        const Status s = cfg.validate();
        if (!s.ok()) return Result<std::unique_ptr<IEngine>>::Err(s);
    }

    const auto* desc = find_descriptor_(cfg.engine);
    if (!desc)
        return Result<std::unique_ptr<IEngine>>::Err(Status::Unsupported("engine_factory: unsupported EngineKind"));

    if (desc->task != cfg.task) {
        return Result<std::unique_ptr<IEngine>>::Err(
            Status::Invalid(std::string("engine_factory: ") + desc->name + " requires a different Task"));
    }

    if (desc->validate) {
        const Status es = desc->validate(cfg);
        if (!es.ok()) return Result<std::unique_ptr<IEngine>>::Err(es);
    }

    try {
        return desc->construct(cfg);
    } catch (const std::bad_alloc&) {
        return Result<std::unique_ptr<IEngine>>::Err(Status::OutOfMemory("engine_factory: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::unique_ptr<IEngine>>::Err(
            Status::Internal(std::string("engine_factory: ctor threw: ") + e.what()));
    } catch (...) {
        return Result<std::unique_ptr<IEngine>>::Err(Status::Internal("engine_factory: ctor threw (unknown)"));
    }
}

} // namespace idet::engine
