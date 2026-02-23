/**
 * @file engine_factory.cpp
 * @ingroup idet_engine
 * @brief Engine factory implementation (DBNet/SCRFD selection).
 *
 * @details
 * Implements @ref idet::engine::create_engine:
 * - validates @ref idet::DetectorConfig,
 * - enforces explicit mapping between @ref idet::EngineKind and @ref idet::Task,
 * - constructs a concrete engine and returns ownership via Result<std::unique_ptr<IEngine>>.
 *
 * Exception safety:
 * - The factory is noexcept and converts any exceptions into @ref idet::Status.
 */

#include "engine/engine_factory.h"

#include "engine/dbnet.h"
#include "engine/scrfd.h"

#include <exception>
#include <new>
#include <string>
#include <utility>

namespace idet::engine {

/**
 * @brief Construct a concrete engine implementation based on @ref idet::DetectorConfig.
 *
 * @details
 * Processing steps:
 * 1) Validate the configuration using @ref idet::DetectorConfig::validate().
 * 2) Select the concrete backend by @ref idet::DetectorConfig::engine.
 * 3) Verify that @ref idet::DetectorConfig::task matches the engine kind.
 * 4) Construct the engine and return it as `std::unique_ptr<IEngine>`.
 *
 * Error handling:
 * - This function is `noexcept`. All exceptions are caught and translated into
 *   @ref idet::Status to avoid exceptions crossing library boundaries.
 *
 * @param cfg Detector configuration. Must specify compatible @ref idet::Task and @ref idet::EngineKind.
 * @return Result::Ok(engine) on success; Result::Err(status) on failure.
 *
 * @retval Status::Invalid If configuration is invalid (including task/engine mismatch).
 * @retval Status::Unsupported If the requested engine kind is not supported by this build.
 * @retval Status::OutOfMemory If allocation fails.
 * @retval Status::Internal If the engine constructor throws an exception.
 *
 * @note
 * Concrete engine constructors are expected to initialize ORT sessions and any required
 * resources (or be ready for inference immediately after construction).
 */
Result<std::unique_ptr<IEngine>> create_engine(const DetectorConfig& cfg) noexcept {
    {
        const Status s = cfg.validate();
        if (!s.ok()) return Result<std::unique_ptr<IEngine>>::Err(s);
    }

    try {
        switch (cfg.engine) {
        case EngineKind::DBNet: {
            if (cfg.task != Task::Text) {
                return Result<std::unique_ptr<IEngine>>::Err(
                    Status::Invalid("engine_factory: DBNet supports only Task::Text"));
            }
            std::unique_ptr<IEngine> p(new DBNet(cfg));
            return Result<std::unique_ptr<IEngine>>::Ok(std::move(p));
        }

        case EngineKind::SCRFD: {
            if (cfg.task != Task::Face) {
                return Result<std::unique_ptr<IEngine>>::Err(
                    Status::Invalid("engine_factory: SCRFD supports only Task::Face"));
            }
            std::unique_ptr<IEngine> p(new SCRFD(cfg));
            return Result<std::unique_ptr<IEngine>>::Ok(std::move(p));
        }

        default:
            return Result<std::unique_ptr<IEngine>>::Err(Status::Unsupported("engine_factory: unsupported EngineKind"));
        }
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
