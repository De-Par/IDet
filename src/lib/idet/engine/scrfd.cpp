/**
 * @file scrfd.cpp
 * @ingroup idet_engine
 * @brief SCRFD engine implementation (layout probing, decoding, and ORT binding).
 *
 * @details
 * This translation unit implements @ref idet::engine::SCRFD:
 * - CHW float32 preprocessing from BGR input,
 * - unbound inference (per-call tensors) via Ort::Session::Run,
 * - bound inference via Ort::IoBinding and preallocated I/O buffers,
 * - robust probing of SCRFD output tensor layouts/shapes across common export variants,
 * - decoding of multi-stride heads (typically 8/16/32) into @ref idet::algo::Detection.
 *
 * Layout variability handling:
 * SCRFD exports differ across toolchains/opsets. For each head, this implementation infers:
 * - score layout: CHW / Flat / HW
 * - bbox layout:  CHW / Flat / HW4
 * and then decodes them with layout-aware accessors.
 *
 * Thread-safety contract:
 * - Unbound mode is safe for concurrent calls.
 * - Bound mode is safe only if each concurrent caller uses a unique context index.
 */

#include "engine/scrfd.h"

#include "internal/chw_preprocess.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

namespace idet::engine {

namespace {

/**
 * @brief Aligns @p v up to the next multiple of @p a.
 *
 * @note
 * Used to match typical SCRFD export constraints (often multiples of 32).
 */
static inline int align_up_(int v, int a) noexcept {
    if (a <= 1) return v;
    return (v + a - 1) / a * a;
}

/**
 * @brief Clamp float value to [lo; hi].
 */
static inline float clampf_(float v, float lo, float hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

} // namespace

/**
 * @brief Construct SCRFD engine: validate config, create ORT session, and initialize metadata.
 *
 * @details
 * Sequence:
 * 1) Validate @ref idet::DetectorConfig and enforce task/engine invariants.
 * 2) Create ORT session (model path or embedded blob) via @ref IEngine::create_session_.
 * 3) Query input/output node names from ORT metadata.
 * 4) Cache hot inference parameters for fast-path decoding.
 *
 * @throws std::runtime_error on validation/session creation failure.
 *         The engine factory wrapper is expected to catch and convert to Status.
 */
SCRFD::SCRFD(const DetectorConfig& cfg) : IEngine(cfg, "idet-scrfd") {
    const Status vs = cfg_.validate();
    if (!vs.ok()) throw std::runtime_error(vs.message);

    if (cfg_.task != Task::Face) throw std::runtime_error("SCRFD: cfg.task must be Face");
    if (cfg_.engine != EngineKind::SCRFD) throw std::runtime_error("SCRFD: cfg.engine must be SCRFD");

    auto s = create_session_(cfg_.model_path, cfg_.engine);
    if (!s.ok()) throw std::runtime_error(s.message);

    init_io_names_();
    cache_hot_();
}

/**
 * @brief Cache "hot" inference parameters from cfg_ into POD members.
 *
 * @details
 * These fields are read frequently inside decode_ and are copied once to avoid repeatedly
 * dereferencing cfg_.infer on the hot path.
 *
 * @note
 * In this engine, @ref idet::InferParams::box_thresh is interpreted as the SCRFD score threshold.
 */
void SCRFD::cache_hot_() noexcept {
    apply_sigmoid_ = cfg_.infer.apply_sigmoid;
    score_thr_ = cfg_.infer.box_thresh;
    max_img_ = cfg_.infer.max_img_size;
    min_w_ = cfg_.infer.min_roi_size_w;
    min_h_ = cfg_.infer.min_roi_size_h;
}

Status SCRFD::update_hot(const DetectorConfig& next) noexcept {
    const Status chk = check_hot_update_(next);
    if (!chk.ok()) return chk;

    apply_hot_common_(next);
    cache_hot_();
    return Status::Ok();
}

/**
 * @brief Resolve ORT input/output node names.
 *
 * @details
 * Uses ORT metadata (GetInputNameAllocated/GetOutputNameAllocated).
 * If name resolution fails, falls back to stable synthetic names ("input", "out_i").
 *
 * @note
 * Names are used to bind tensors in bound mode via Ort::IoBinding.
 */
void SCRFD::init_io_names_() {
    Ort::AllocatedStringPtr in0 = session_.GetInputNameAllocated(0, alloc_);
    in_name_ = in0 ? in0.get() : std::string("input");

    const std::size_t nout = session_.GetOutputCount();
    out_names_.clear();
    out_names_.reserve(nout);
    for (std::size_t i = 0; i < nout; ++i) {
        Ort::AllocatedStringPtr on = session_.GetOutputNameAllocated(i, alloc_);
        out_names_.push_back(on ? on.get() : ("out_" + std::to_string(i)));
    }
}

inline float SCRFD::sigmoid_(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

algo::Detection SCRFD::rect_to_det_(float x1, float y1, float x2, float y2, float score) {
    algo::Detection d;
    d.score = score;
    d.pts[0] = cv::Point2f(x1, y1);
    d.pts[1] = cv::Point2f(x2, y1);
    d.pts[2] = cv::Point2f(x2, y2);
    d.pts[3] = cv::Point2f(x1, y2);
    return d;
}

/**
 * @brief Fill CHW float32 input buffer for SCRFD.
 *
 * @details
 * SCRFD commonly uses normalization: (x - 127.5) / 128 for each channel.
 *
 * @param dst Destination buffer in CHW order (size = 3 * in_h * in_w).
 * @param in_w/in_h Target network input dimensions (already aligned if needed).
 * @param bgr Source image in BGR order (CV_8UC3).
 */
void SCRFD::fill_input_chw_(float* dst, int in_w, int in_h, const cv::Mat& bgr) const {
    // SCRFD: (x - 127.5) / 128
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float inv_std[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
    internal::bgr_u8_to_chw_f32_resize(bgr, in_w, in_h, dst, mean, inv_std);
}

/**
 * @brief Run SCRFD in unbound mode and return raw ORT outputs.
 *
 * @details
 * Performs:
 * - input resize to a (possibly clamped) size, aligned to 32,
 * - CHW float32 preprocessing with SCRFD normalization,
 * - Ort::Session::Run producing all model outputs.
 *
 * Geometry:
 * - sx = in_w / orig_w, sy = in_h / orig_h are returned to map decoded boxes back
 *   to original image coordinates in @ref decode_.
 *
 * @param bgr Input image (BGR, CV_8UC3).
 * @param force_w/force_h If both > 0, force a fixed input shape (still aligned to 32).
 * @param sx/sy Output scale factors (network / original).
 * @param in_w/in_h Output effective network input shape.
 * @return Vector of Ort::Value outputs in the same order as @ref out_names_.
 */
Result<std::vector<Ort::Value>> SCRFD::run_unbound_(const cv::Mat& bgr, int force_w, int force_h, float& sx, float& sy,
                                                    int& in_w, int& in_h) noexcept {
    try {
        if (bgr.empty() || bgr.type() != CV_8UC3) {
            return Result<std::vector<Ort::Value>>::Err(Status::Invalid("SCRFD: run_unbound expects CV_8UC3 BGR"));
        }

        const int ow = bgr.cols;
        const int oh = bgr.rows;

        int tw = force_w;
        int th = force_h;
        if (tw <= 0 || th <= 0) {
            tw = ow;
            th = oh;
            if (max_img_ > 0) {
                const int max_side = std::max(ow, oh);
                if (max_side > max_img_) {
                    const float scale = (float)max_img_ / (float)max_side;
                    tw = std::max(1, (int)std::lround(ow * scale));
                    th = std::max(1, (int)std::lround(oh * scale));
                }
            }
        }
        tw = align_up_(tw, 32);
        th = align_up_(th, 32);

        in_w = tw;
        in_h = th;

        sx = (float)tw / (float)ow;
        sy = (float)th / (float)oh;

        std::vector<float> chw((std::size_t)3 * (std::size_t)th * (std::size_t)tw);
        fill_input_chw_(chw.data(), tw, th, bgr);

        static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> ishape = {1, 3, in_h, in_w};

        Ort::Value in_tensor =
            Ort::Value::CreateTensor<float>(cpu_mem, chw.data(), chw.size(), ishape.data(), ishape.size());

        std::vector<const char*> out_names_c;
        out_names_c.reserve(out_names_.size());
        for (auto& s : out_names_)
            out_names_c.push_back(s.c_str());

        const char* in_names[] = {in_name_.c_str()};

        auto outs =
            session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names_c.data(), out_names_c.size());

        return Result<std::vector<Ort::Value>>::Ok(std::move(outs));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<Ort::Value>>::Err(Status::OutOfMemory("SCRFD: run_unbound bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<Ort::Value>>::Err(Status::Internal(std::string("SCRFD: run_unbound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<Ort::Value>>::Err(Status::Internal("SCRFD: run_unbound: unknown"));
    }
}

/**
 * @brief Probe model outputs and infer per-head tensor layouts.
 *
 * @details
 * The method runs a dummy inference at a fixed input shape and then:
 * - resolves output indices for each stride head (8/16/32) by matching output names
 *   against common substrings ("score"/"bbox"/"cls"/"conf"/"reg") and stride token,
 * - falls back to a conventional 6-output ordering if name matching fails,
 * - reads tensor shapes from ORT and infers layouts:
 *   - score: CHW ([1,C,H,W]) / Flat ([1,N,C] or [1,N,1]) / HW ([H,W] or [1,H,W])
 *   - bbox : CHW ([1,4,H,W]) / Flat ([1,N,4]) / HW4 ([H,W,4] or [1,H,W,4])
 * - infers anchors for flat exports by checking N % (Hs*Ws) == 0.
 *
 * @param in_h/in_w Effective network input shape (already aligned).
 * @param heads Output vector filled with valid heads; inconsistent heads are skipped.
 *
 * @retval Status::Unsupported if no consistent heads could be resolved.
 */
Status SCRFD::probe_heads_layout_(int in_h, int in_w, std::vector<Head>* heads) noexcept {
    try {
        cv::Mat dummy(in_h, in_w, CV_8UC3, cv::Scalar(0, 0, 0));

        float sx = 1.f, sy = 1.f;
        int iw = 0, ih = 0;
        auto r = run_unbound_(dummy, in_w, in_h, sx, sy, iw, ih);
        if (!r.ok()) return r.status();

        auto outs = std::move(r.value());
        if (outs.size() != out_names_.size()) return Status::Internal("SCRFD: probe outputs count mismatch");

        // Name-based best-effort matching: robust across exporters that keep semantic tokens.
        auto find_by = [&](const std::string& what, const std::string& stride) -> int {
            for (int i = 0; i < (int)out_names_.size(); ++i) {
                const auto& n = out_names_[(std::size_t)i];
                if (n.find(what) != std::string::npos && n.find(stride) != std::string::npos) return i;
            }
            return -1;
        };

        std::vector<Head> hs;
        hs.reserve(3);

        auto infer_score_layout = [&](const std::vector<int64_t>& sshape, Head& h) {
            // common: [1,C,H,W] or [1,N,C] or [1,H,W]
            if (sshape.size() == 4) {
                // treat as CHW if second dim is small channel
                if (sshape[1] > 0 && sshape[1] <= 8) {
                    h.score_layout = Layout::Score_CHW;
                    h.score_ch = (int)std::max<int64_t>(1, sshape[1]);
                    h.Hs = (int)sshape[2];
                    h.Ws = (int)sshape[3];
                } else {
                    // fallback: [1,H,W,C] like exports
                    h.score_layout = Layout::Score_HW;
                    h.Hs = (int)sshape[1];
                    h.Ws = (int)sshape[2];
                    h.score_ch = (sshape.size() >= 4) ? (int)std::max<int64_t>(1, sshape[3]) : 1;
                }
            } else if (sshape.size() == 3) {
                // [1,N,C] or [1,H,W]
                if (sshape[0] == 1 && sshape[2] > 0 && sshape[2] <= 8) {
                    h.score_layout = Layout::Score_Flat;
                    h.score_ch = (int)std::max<int64_t>(1, sshape[2]);
                } else {
                    h.score_layout = Layout::Score_HW;
                    h.Hs = (int)sshape[1];
                    h.Ws = (int)sshape[2];
                    h.score_ch = 1;
                }
            }
        };

        auto infer_bbox_layout = [&](const std::vector<int64_t>& bshape, Head& h) {
            // common: [1,4,H,W] or [1,N,4] or [1,H,W,4]
            if (bshape.size() == 4 && bshape[1] == 4) {
                h.bbox_layout = Layout::BBox_CHW;
                h.Hs = (int)bshape[2];
                h.Ws = (int)bshape[3];
            } else if (bshape.size() == 3 && bshape[2] == 4) {
                h.bbox_layout = Layout::BBox_Flat;
            } else if (bshape.size() == 4 && bshape[3] == 4) {
                h.bbox_layout = Layout::BBox_HW4;
                h.Hs = (int)bshape[1];
                h.Ws = (int)bshape[2];
            }
        };

        auto add_head = [&](int stride) {
            Head h;
            h.stride = stride;

            const std::string s = std::to_string(stride);

            int si = find_by("score", s);
            int bi = find_by("bbox", s);

            if (si < 0) si = find_by("cls", s);
            if (si < 0) si = find_by("conf", s);
            if (bi < 0) bi = find_by("reg", s);

            if (si < 0 || bi < 0) {
                // Fallback for common exports with fixed output ordering:
                // (score8, score16, score32, bbox8, bbox16, bbox32).
                if (out_names_.size() >= 6) {
                    si = (stride == 8) ? 0 : (stride == 16 ? 1 : 2);
                    bi = (stride == 8) ? 3 : (stride == 16 ? 4 : 5);
                } else {
                    return;
                }
            }

            h.score_idx = si;
            h.bbox_idx = bi;

            h.score_shape = outs[(std::size_t)si].GetTensorTypeAndShapeInfo().GetShape();
            h.bbox_shape = outs[(std::size_t)bi].GetTensorTypeAndShapeInfo().GetShape();

            // base guess
            h.Hs = std::max(1, in_h / stride);
            h.Ws = std::max(1, in_w / stride);
            h.anchors = 1;
            h.score_ch = 1;

            infer_score_layout(h.score_shape, h);
            infer_bbox_layout(h.bbox_shape, h);

            // infer anchors for flat exports
            if (h.score_layout == Layout::Score_Flat && h.score_shape.size() == 3) {
                const int64_t Nloc = h.score_shape[1];
                const int hw = std::max(1, h.Hs * h.Ws);
                if (hw > 0 && Nloc % hw == 0) h.anchors = (int)(Nloc / hw);
            }
            if (h.bbox_layout == Layout::BBox_Flat && h.bbox_shape.size() == 3) {
                const int64_t Nloc = h.bbox_shape[1];
                const int hw = std::max(1, h.Hs * h.Ws);
                if (hw > 0 && Nloc % hw == 0) h.anchors = (int)(Nloc / hw);
            }

            if (h.score_layout == Layout::Unknown || h.bbox_layout == Layout::Unknown) {
                return; // skip inconsistent head
            }

            hs.push_back(std::move(h));
        };

        add_head(8);
        add_head(16);
        add_head(32);

        if (hs.empty()) return Status::Unsupported("SCRFD: cannot resolve heads");
        *heads = std::move(hs);
        return Status::Ok();
    } catch (const std::exception& e) {
        return Status::Internal(std::string("SCRFD: probe_heads_layout: ") + e.what());
    } catch (...) {
        return Status::Internal("SCRFD: probe_heads_layout: unknown");
    }
}

/**
 * @brief Decode per-head SCRFD outputs into detections.
 *
 * @details
 * For each head (stride 8/16/32), iterate over all spatial locations and anchors:
 * - read score using the inferred score layout,
 * - optionally apply sigmoid,
 * - read bbox distances/coords using inferred bbox layout,
 * - convert (dl,dt,dr,db) and center point to (x1,y1,x2,y2),
 * - scale back to original image coordinates using (sx, sy),
 * - clamp to image bounds and apply min size filtering.
 *
 * @note Channel selection:
 * This implementation uses a fixed channel choice for multi-channel score outputs
 * (currently: channel 1 if score_ch>1 else 0). If your exports differ (e.g. face class at ch=0),
 * this should be made configurable in the config/infer params.
 */
std::vector<algo::Detection> SCRFD::decode_(const std::vector<Head>& heads, const std::vector<const float*>& score_ptrs,
                                            const std::vector<const float*>& bbox_ptrs, float sx, float sy, int orig_w,
                                            int orig_h) const {
    std::vector<algo::Detection> dets;
    dets.reserve(256);

    for (std::size_t hi = 0; hi < heads.size(); ++hi) {
        const auto& h = heads[hi];
        const float* score = score_ptrs[hi];
        const float* bbox = bbox_ptrs[hi];
        if (!score || !bbox) continue;

        const int Hs = std::max(1, h.Hs);
        const int Ws = std::max(1, h.Ws);
        const int A = std::max(1, h.anchors);
        const int stride = h.stride;
        const int hw = Hs * Ws;

        auto score_at = [&](int y, int x, int a) -> float {
            // production default: take channel 0 (или "face" во втором канале — это лучше параметризовать)
            const int ch = (h.score_ch > 1) ? 1 : 0;

            if (h.score_layout == Layout::Score_CHW) {
                return score[ch * hw + (y * Ws + x)];
            }
            if (h.score_layout == Layout::Score_Flat) {
                const int loc = (y * Ws + x) * A + a;
                return score[loc * h.score_ch + ch];
            }
            // Score_HW
            return score[y * Ws + x];
        };

        auto bbox_at = [&](int y, int x, int a, float& dl, float& dt, float& dr, float& db) {
            if (h.bbox_layout == Layout::BBox_CHW) {
                const int idx = (y * Ws + x);
                dl = bbox[0 * hw + idx] * stride;
                dt = bbox[1 * hw + idx] * stride;
                dr = bbox[2 * hw + idx] * stride;
                db = bbox[3 * hw + idx] * stride;
                return;
            }
            if (h.bbox_layout == Layout::BBox_Flat) {
                const int loc = (y * Ws + x) * A + a;
                dl = bbox[loc * 4 + 0] * stride;
                dt = bbox[loc * 4 + 1] * stride;
                dr = bbox[loc * 4 + 2] * stride;
                db = bbox[loc * 4 + 3] * stride;
                return;
            }
            // BBox_HW4: [H,W,4] contiguous
            const int idx = (y * Ws + x) * 4;
            dl = bbox[idx + 0] * stride;
            dt = bbox[idx + 1] * stride;
            dr = bbox[idx + 2] * stride;
            db = bbox[idx + 3] * stride;
        };

        for (int y = 0; y < Hs; ++y) {
            for (int x = 0; x < Ws; ++x) {
                for (int a = 0; a < A; ++a) {
                    float sc = score_at(y, x, a);
                    if (apply_sigmoid_) sc = sigmoid_(sc);
                    if (sc < score_thr_) continue;

                    float dl = 0, dt = 0, dr = 0, db = 0;
                    bbox_at(y, x, a, dl, dt, dr, db);

                    const float cx = (x + 0.5f) * stride;
                    const float cy = (y + 0.5f) * stride;

                    float x1 = (cx - dl) / sx;
                    float y1 = (cy - dt) / sy;
                    float x2 = (cx + dr) / sx;
                    float y2 = (cy + db) / sy;

                    x1 = clampf_(x1, 0.f, (float)orig_w);
                    y1 = clampf_(y1, 0.f, (float)orig_h);
                    x2 = clampf_(x2, 0.f, (float)orig_w);
                    y2 = clampf_(y2, 0.f, (float)orig_h);

                    if (x2 <= x1 || y2 <= y1) continue;
                    if (min_w_ > 0 && (x2 - x1) < (float)min_w_) continue;
                    if (min_h_ > 0 && (y2 - y1) < (float)min_h_) continue;

                    dets.push_back(rect_to_det_(x1, y1, x2, y2, sc));
                }
            }
        }
    }

    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    return dets;
}

/**
 * @brief Prepare bound inference: preallocate per-context I/O and bind outputs by name.
 *
 * @details
 * - Input shape is aligned to 32 and fixed for all subsequent bound calls.
 * - Heads and output indices are probed once and then frozen in @ref heads_ and @ref bound_out_indices_.
 * - Each context allocates:
 *   - input CHW buffer,
 *   - output buffers for [score,bbox,...] in @ref bound_out_indices_ order,
 *   - Ort::Value tensors wrapping these buffers,
 *   - Ort::IoBinding bindings for fast Session::Run.
 *
 * Concurrency:
 * Each context must be used by at most one concurrent caller.
 */
Status SCRFD::setup_binding(int w, int h, int contexts) noexcept {
    try {
        unset_binding();

        if (w <= 0 || h <= 0) return Status::Invalid("SCRFD::setup_binding: non-positive w/h");
        if (contexts <= 0) contexts = 1;

        bound_w_ = w;
        bound_h_ = h;
        contexts_ = contexts;

        const int in_w = align_up_(w, 32);
        const int in_h = align_up_(h, 32);

        std::vector<Head> heads;
        Status ps = probe_heads_layout_(in_h, in_w, &heads);
        if (!ps.ok()) return ps;

        heads_ = std::move(heads);

        bound_out_indices_.clear();
        bound_out_indices_.reserve(heads_.size() * 2);
        for (auto& hd : heads_) {
            bound_out_indices_.push_back(hd.score_idx);
            bound_out_indices_.push_back(hd.bbox_idx);
        }

        static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        ctxs_.resize((std::size_t)contexts_);
        for (int ci = 0; ci < contexts_; ++ci) {
            auto& c = ctxs_[(std::size_t)ci];

            c.binding = std::make_unique<Ort::IoBinding>(session_);

            c.in.assign((std::size_t)3 * (std::size_t)in_h * (std::size_t)in_w, 0.f);
            std::vector<int64_t> ishape = {1, 3, in_h, in_w};
            c.in_tensor =
                Ort::Value::CreateTensor<float>(cpu_mem, c.in.data(), c.in.size(), ishape.data(), ishape.size());
            c.binding->BindInput(in_name_.c_str(), c.in_tensor);

            c.outs.clear();
            c.out_tensors.clear();
            c.outs.resize(bound_out_indices_.size());
            c.out_tensors.reserve(bound_out_indices_.size());

            for (std::size_t oi = 0; oi < bound_out_indices_.size(); ++oi) {
                const bool is_score = (oi % 2 == 0);
                const Head& hd = heads_[(std::size_t)(oi / 2)];
                const int out_idx = bound_out_indices_[oi];

                const auto& shape = is_score ? hd.score_shape : hd.bbox_shape;

                std::size_t numel = 1;
                for (auto v : shape)
                    numel *= (std::size_t)std::max<int64_t>(1, v);

                c.outs[oi].assign(numel, 0.f);
                c.out_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                    cpu_mem, c.outs[oi].data(), c.outs[oi].size(), shape.data(), shape.size()));

                c.binding->BindOutput(out_names_[(std::size_t)out_idx].c_str(), c.out_tensors.back());
            }
        }

        binding_ready_ = true;
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        unset_binding();
        return Status::OutOfMemory("SCRFD::setup_binding: bad_alloc");
    } catch (const std::exception& e) {
        unset_binding();
        return Status::Internal(std::string("SCRFD::setup_binding: ") + e.what());
    } catch (...) {
        unset_binding();
        return Status::Internal("SCRFD::setup_binding: unknown");
    }
}

void SCRFD::unset_binding() noexcept {
    binding_ready_ = false;
    bound_w_ = bound_h_ = 0;
    contexts_ = 0;
    ctxs_.clear();
    heads_.clear();
    bound_out_indices_.clear();
}

/**
 * @brief Unbound inference: run ORT and decode to detections.
 *
 * @details
 * Probes heads lazily on first call if @ref heads_ is empty (export-dependent).
 */
Result<std::vector<algo::Detection>> SCRFD::infer_unbound(const cv::Mat& bgr) noexcept {
    try {
        if (bgr.empty() || bgr.type() != CV_8UC3) {
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("SCRFD::infer_unbound: expected CV_8UC3 BGR"));
        }

        float sx = 1.f, sy = 1.f;
        int in_w = 0, in_h = 0;
        auto rr = run_unbound_(bgr, 0, 0, sx, sy, in_w, in_h);
        if (!rr.ok()) return Result<std::vector<algo::Detection>>::Err(rr.status());

        auto outs = std::move(rr.value());
        if (outs.size() != out_names_.size()) {
            return Result<std::vector<algo::Detection>>::Err(Status::Internal("SCRFD: outputs count mismatch"));
        }

        if (heads_.empty()) {
            std::vector<Head> hs;
            Status ps = probe_heads_layout_(in_h, in_w, &hs);
            if (!ps.ok()) return Result<std::vector<algo::Detection>>::Err(ps);
            heads_ = std::move(hs);
        }

        std::vector<const float*> score_ptrs(heads_.size(), nullptr);
        std::vector<const float*> bbox_ptrs(heads_.size(), nullptr);

        for (std::size_t hi = 0; hi < heads_.size(); ++hi) {
            const Head& hd = heads_[hi];
            if (hd.score_idx < 0 || hd.bbox_idx < 0) continue;

            score_ptrs[hi] = outs[(std::size_t)hd.score_idx].GetTensorData<float>();
            bbox_ptrs[hi] = outs[(std::size_t)hd.bbox_idx].GetTensorData<float>();
        }

        auto dets = decode_(heads_, score_ptrs, bbox_ptrs, sx, sy, bgr.cols, bgr.rows);
        return Result<std::vector<algo::Detection>>::Ok(std::move(dets));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<algo::Detection>>::Err(Status::OutOfMemory("SCRFD::infer_unbound: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<algo::Detection>>::Err(
            Status::Internal(std::string("SCRFD::infer_unbound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<algo::Detection>>::Err(Status::Internal("SCRFD::infer_unbound: unknown"));
    }
}

/**
 * @brief Bound inference: uses preallocated per-context buffers and IoBinding.
 *
 * @pre @ref binding_ready() is true and ctx_idx is valid.
 * @note The implementation assumes the bound input shape is (align_up(bound_w_,32), align_up(bound_h_,32)).
 */
Result<std::vector<algo::Detection>> SCRFD::infer_bound(const cv::Mat& bgr, int ctx_idx) noexcept {
    try {
        if (!binding_ready_)
            return Result<std::vector<algo::Detection>>::Err(Status::Invalid("SCRFD::infer_bound: binding not ready"));
        if (ctx_idx < 0 || ctx_idx >= contexts_)
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("SCRFD::infer_bound: ctx_idx out of range"));
        if (bgr.empty() || bgr.type() != CV_8UC3)
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("SCRFD::infer_bound: expected CV_8UC3 BGR"));

        auto& c = ctxs_[(std::size_t)ctx_idx];

        const int in_w = align_up_(bound_w_, 32);
        const int in_h = align_up_(bound_h_, 32);

        const float sx = (float)in_w / (float)bgr.cols;
        const float sy = (float)in_h / (float)bgr.rows;

        const float mean[3] = {127.5f, 127.5f, 127.5f};
        const float inv_std[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
        internal::bgr_u8_to_chw_f32_resize(bgr, in_w, in_h, c.in.data(), mean, inv_std);

        session_.Run(Ort::RunOptions{nullptr}, *c.binding);

        std::vector<const float*> score_ptrs(heads_.size(), nullptr);
        std::vector<const float*> bbox_ptrs(heads_.size(), nullptr);

        for (std::size_t hi = 0; hi < heads_.size(); ++hi) {
            const std::size_t score_i = hi * 2 + 0;
            const std::size_t bbox_i = hi * 2 + 1;
            score_ptrs[hi] = (score_i < c.outs.size()) ? c.outs[score_i].data() : nullptr;
            bbox_ptrs[hi] = (bbox_i < c.outs.size()) ? c.outs[bbox_i].data() : nullptr;
        }

        auto dets = decode_(heads_, score_ptrs, bbox_ptrs, sx, sy, bgr.cols, bgr.rows);
        return Result<std::vector<algo::Detection>>::Ok(std::move(dets));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<algo::Detection>>::Err(Status::OutOfMemory("SCRFD::infer_bound: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<algo::Detection>>::Err(
            Status::Internal(std::string("SCRFD::infer_bound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<algo::Detection>>::Err(Status::Internal("SCRFD::infer_bound: unknown"));
    }
}

} // namespace idet::engine
