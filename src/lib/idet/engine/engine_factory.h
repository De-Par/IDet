/**
 * @file engine_factory.h
 * @ingroup idet_engine
 * @brief Factory function for creating a concrete ORT-based engine implementation.
 *
 * @details
 * This header declares the engine factory used by the public detector facade.
 * The factory examines the provided @ref idet::DetectorConfig and instantiates a
 * concrete @ref idet::engine::IEngine implementation.
 *
 * Engine selection rules (high-level):
 * - Text task / DBNet engine  -> DBNet backend
 * - Face task / SCRFD engine  -> SCRFD backend
 *
 * The factory is responsible for basic configuration validation and for returning
 * a structured error via @ref idet::Status / @ref idet::Result instead of throwing.
 */

#pragma once

#include "engine/engine.h"
#include "status.h"

#include <memory>

namespace idet::engine {

/**
 * @brief Create a concrete engine instance according to the provided configuration.
 *
 * @details
 * This function validates the configuration and constructs a concrete engine that
 * implements @ref idet::engine::IEngine. The concrete backend is chosen based on
 * @ref idet::DetectorConfig::task and @ref idet::DetectorConfig::engine.
 *
 * On success, the returned `std::unique_ptr<IEngine>` owns the created engine object.
 * The caller receives exclusive ownership and is responsible for keeping the instance
 * alive for as long as it is used.
 *
 * Error handling:
 * - No exceptions are expected to escape this function; failures are represented as
 *   a non-OK @ref idet::Status inside the returned @ref idet::Result.
 *
 * @param cfg Detector configuration. Must be internally consistent (task/engine match)
 *            and contain a valid model source (implementation-defined: filesystem path
 *            and/or embedded model availability).
 *
 * @return `Result<std::unique_ptr<IEngine>>`:
 * - `ok() == true`  -> owns a fully constructed engine instance
 * - `ok() == false` -> contains an error @ref idet::Status describing the reason
 *
 * @note
 * The concrete engine constructor may create ORT sessions and allocate resources.
 * Callers should ensure that any required process-wide runtime policy (CPU affinity,
 * OpenMP configuration, etc.) has been applied before creating engines, if such
 * policies are part of their performance/latency contract.
 */
Result<std::unique_ptr<IEngine>> create_engine(const DetectorConfig& cfg) noexcept;

} // namespace idet::engine
