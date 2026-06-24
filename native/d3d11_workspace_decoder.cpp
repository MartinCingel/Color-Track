// Windows-native foundation for the vendor-neutral workspace decoder.
// The exported ABI is intentionally small so Python can load the DLL with
// ctypes and future installer builds do not depend on a Python C extension.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>
#include <wrl/client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/d3d11va.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {

void WriteMessage(char* output, unsigned int capacity, const std::string& message) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(output, capacity, "%s", message.c_str());
    output[capacity - 1] = '\0';
}

std::string Utf8FromWide(const wchar_t* value) {
    if (value == nullptr || value[0] == L'\0') return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<size_t>(size > 0 ? size - 1 : 0), '\0');
    if (size > 1) {
        WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr, nullptr);
    }
    return result;
}

std::string AvError(int error) {
    char buffer[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(error, buffer, sizeof(buffer));
    return buffer;
}

enum AVPixelFormat GetD3D11Format(AVCodecContext*, const enum AVPixelFormat* formats) {
    for (const enum AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_D3D11) return *format;
    }
    return AV_PIX_FMT_NONE;
}

const char* VendorName(UINT vendor_id) {
    switch (vendor_id) {
    case 0x10DE: return "NVIDIA";
    case 0x1002: return "AMD";
    case 0x8086: return "Intel";
    default: return "Unknown";
    }
}

HRESULT CreateVideoDevice(ComPtr<ID3D11Device>* device_out, std::string* adapter_name) {
    ComPtr<IDXGIFactory1> factory;
    HRESULT result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(result)) return result;
    for (UINT index = 0; ; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) return result;
        if (FAILED(result)) continue;
        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description)) || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected{};
        ComPtr<ID3D11DeviceContext> context;
        ComPtr<ID3D11Device> device;
        result = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION, &device, &selected, &context);
        if (FAILED(result)) continue;
        if (adapter_name != nullptr) {
            char name[256]{};
            WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name,
                static_cast<int>(sizeof(name)), nullptr, nullptr);
            *adapter_name = name;
        }
        *device_out = device;
        return S_OK;
    }
}

}  // namespace

extern "C" __declspec(dllexport) unsigned int ct_d3d11_abi_version() {
    return 1;
}

// Returns 0 when a hardware D3D11 video device is available. The message
// describes the first suitable adapter and is safe to display in the UI.
extern "C" __declspec(dllexport) int ct_d3d11_probe(char* output, unsigned int capacity) {
    HRESULT co_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool should_uninitialize = SUCCEEDED(co_result);

    HRESULT mf_result = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    if (FAILED(mf_result)) {
        WriteMessage(output, capacity, "Media Foundation startup failed.");
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(mf_result);
    }

    ComPtr<IDXGIFactory1> factory;
    HRESULT result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Could not create a DXGI factory.");
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }

    for (UINT index = 0; ; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(result)) continue;

        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description)) || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            continue;
        }

        const D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected{};
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        result = D3D11CreateDevice(
            adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            requested, ARRAYSIZE(requested), D3D11_SDK_VERSION,
            &device, &selected, &context
        );
        if (FAILED(result)) continue;

        ComPtr<ID3D11VideoDevice> video_device;
        result = device.As(&video_device);
        if (FAILED(result)) continue;

        char utf8_name[256]{};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, utf8_name,
                            static_cast<int>(sizeof(utf8_name)), nullptr, nullptr);
        char message[512]{};
        std::snprintf(
            message, sizeof(message),
            "%s D3D11 video device ready: %s (vendor 0x%04X, feature level 0x%X).",
            VendorName(description.VendorId), utf8_name, description.VendorId, selected
        );
        WriteMessage(output, capacity, message);
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return 0;
    }

    WriteMessage(output, capacity, "No hardware D3D11 video device was found.");
    MFShutdown();
    if (should_uninitialize) CoUninitialize();
    return static_cast<int>(DXGI_ERROR_NOT_FOUND);
}

// Opens a video through Media Foundation and verifies that at least one
// decoded sample is backed by an ID3D11Texture2D rather than CPU memory.
extern "C" __declspec(dllexport) int ct_d3d11_decode_probe(
    const wchar_t* video_path, char* output, unsigned int capacity) {
    if (video_path == nullptr || video_path[0] == L'\0') {
        WriteMessage(output, capacity, "No video path was supplied.");
        return E_INVALIDARG;
    }
    HRESULT co_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool should_uninitialize = SUCCEEDED(co_result);
    HRESULT result = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Media Foundation startup failed.");
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }

    ComPtr<ID3D11Device> device;
    std::string adapter_name;
    result = CreateVideoDevice(&device, &adapter_name);
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Could not create a D3D11 video device.");
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }
    UINT reset_token = 0;
    ComPtr<IMFDXGIDeviceManager> manager;
    result = MFCreateDXGIDeviceManager(&reset_token, &manager);
    if (SUCCEEDED(result)) result = manager->ResetDevice(device.Get(), reset_token);
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Could not attach D3D11 device to Media Foundation.");
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }

    ComPtr<IMFAttributes> attributes;
    result = MFCreateAttributes(&attributes, 4);
    if (SUCCEEDED(result)) result = attributes->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, manager.Get());
    if (SUCCEEDED(result)) result = attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Could not configure Media Foundation hardware decode.");
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }

    ComPtr<IMFSourceReader> reader;
    result = MFCreateSourceReaderFromURL(video_path, attributes.Get(), &reader);
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Media Foundation could not open this video.");
        attributes.Reset();
        manager.Reset();
        device.Reset();
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }
    reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
    reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
    ComPtr<IMFMediaType> output_type;
    result = MFCreateMediaType(&output_type);
    if (SUCCEEDED(result)) result = output_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if (SUCCEEDED(result)) result = output_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
    if (SUCCEEDED(result)) result = reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, output_type.Get());
    if (FAILED(result)) {
        WriteMessage(output, capacity, "Hardware NV12 output is unavailable for this video.");
        output_type.Reset();
        reader.Reset();
        attributes.Reset();
        manager.Reset();
        device.Reset();
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return static_cast<int>(result);
    }

    bool found_texture = false;
    std::string success_message;
    for (int attempts = 0; attempts < 64; ++attempts) {
        DWORD flags = 0;
        ComPtr<IMFSample> sample;
        result = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, &flags, nullptr, &sample);
        if (FAILED(result) || (flags & MF_SOURCE_READERF_ENDOFSTREAM)) break;
        if (!sample) continue;
        ComPtr<IMFMediaBuffer> buffer;
        if (FAILED(sample->GetBufferByIndex(0, &buffer))) continue;
        ComPtr<IMFDXGIBuffer> dxgi_buffer;
        if (SUCCEEDED(buffer.As(&dxgi_buffer))) {
            ComPtr<ID3D11Texture2D> texture;
            if (SUCCEEDED(dxgi_buffer->GetResource(IID_PPV_ARGS(&texture)))) {
                D3D11_TEXTURE2D_DESC description{};
                texture->GetDesc(&description);
                char message[512]{};
                std::snprintf(message, sizeof(message),
                    "Media Foundation decoded on %s to D3D11 texture: %ux%u, format %u.",
                    adapter_name.c_str(), description.Width, description.Height, description.Format);
                success_message = message;
                found_texture = true;
                break;
            }
        }
        if (found_texture) break;
    }
    // Media Foundation must outlive every reader/device-manager COM object.
    // Release them explicitly before MFShutdown rather than relying on C++
    // function-scope destruction after the shutdown call.
    output_type.Reset();
    reader.Reset();
    attributes.Reset();
    manager.Reset();
    device.Reset();
    if (found_texture) {
        WriteMessage(output, capacity, success_message);
        MFShutdown();
        if (should_uninitialize) CoUninitialize();
        return 0;
    }
    WriteMessage(output, capacity, "Media Foundation returned CPU-backed frames; no D3D11 decode surface was exposed.");
    MFShutdown();
    if (should_uninitialize) CoUninitialize();
    return E_FAIL;
}

// FFmpeg supplies free H.264, HEVC and AV1 decode support. When D3D11VA is
// available, decoded AVFrames contain an ID3D11Texture2D suitable for the
// shared ROI shader; no Microsoft Store codec extension is required.
extern "C" __declspec(dllexport) int ct_ffmpeg_d3d11_decode_probe(
    const wchar_t* video_path, char* output, unsigned int capacity) {
    const std::string path = Utf8FromWide(video_path);
    if (path.empty()) {
        WriteMessage(output, capacity, "No video path was supplied.");
        return AVERROR(EINVAL);
    }
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVBufferRef* hardware_device = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    int result = avformat_open_input(&format, path.c_str(), nullptr, nullptr);
    if (result < 0) {
        WriteMessage(output, capacity, "FFmpeg could not open this video: " + AvError(result));
        goto cleanup;
    }
    result = avformat_find_stream_info(format, nullptr);
    if (result < 0) {
        WriteMessage(output, capacity, "FFmpeg could not read stream information: " + AvError(result));
        goto cleanup;
    }
    const int stream_index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        result = stream_index;
        WriteMessage(output, capacity, "FFmpeg found no video stream: " + AvError(result));
        goto cleanup;
    }
    const AVCodecParameters* parameters = format->streams[stream_index]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(parameters->codec_id);
    if (decoder == nullptr) {
        result = AVERROR_DECODER_NOT_FOUND;
        WriteMessage(output, capacity, "FFmpeg has no decoder for this video codec.");
        goto cleanup;
    }
    codec = avcodec_alloc_context3(decoder);
    if (codec == nullptr) {
        result = AVERROR(ENOMEM);
        WriteMessage(output, capacity, "FFmpeg could not allocate a decoder context.");
        goto cleanup;
    }
    result = avcodec_parameters_to_context(codec, parameters);
    if (result < 0) {
        WriteMessage(output, capacity, "FFmpeg could not configure the decoder: " + AvError(result));
        goto cleanup;
    }
    result = av_hwdevice_ctx_create(&hardware_device, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (result < 0) {
        WriteMessage(output, capacity, "FFmpeg D3D11VA device is unavailable: " + AvError(result));
        goto cleanup;
    }
    codec->hw_device_ctx = av_buffer_ref(hardware_device);
    codec->get_format = GetD3D11Format;
    result = avcodec_open2(codec, decoder, nullptr);
    if (result < 0) {
        WriteMessage(output, capacity, "FFmpeg could not open a D3D11VA decoder: " + AvError(result));
        goto cleanup;
    }
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (packet == nullptr || frame == nullptr) {
        result = AVERROR(ENOMEM);
        WriteMessage(output, capacity, "FFmpeg could not allocate packet/frame storage.");
        goto cleanup;
    }
    while ((result = av_read_frame(format, packet)) >= 0) {
        if (packet->stream_index != stream_index) {
            av_packet_unref(packet);
            continue;
        }
        result = avcodec_send_packet(codec, packet);
        av_packet_unref(packet);
        if (result < 0 && result != AVERROR(EAGAIN)) break;
        while ((result = avcodec_receive_frame(codec, frame)) >= 0) {
            if (frame->format == AV_PIX_FMT_D3D11) {
                const auto* texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
                const UINT subresource = static_cast<UINT>(reinterpret_cast<intptr_t>(frame->data[1]));
                if (texture != nullptr) {
                    char message[512]{};
                    std::snprintf(message, sizeof(message),
                        "FFmpeg D3D11VA decoded %s to D3D11 texture: %dx%d, subresource %u.",
                        decoder->name, frame->width, frame->height, subresource);
                    WriteMessage(output, capacity, message);
                    result = 0;
                    goto cleanup;
                }
            }
            av_frame_unref(frame);
        }
        if (result != AVERROR(EAGAIN) && result != AVERROR_EOF) break;
    }
    if (result >= 0 || result == AVERROR_EOF) result = AVERROR(ENOSYS);
    WriteMessage(output, capacity, "FFmpeg did not expose a D3D11 hardware frame: " + AvError(result));

cleanup:
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec);
    av_buffer_unref(&hardware_device);
    avformat_close_input(&format);
    return result;
}

namespace {

constexpr char kShaderSource[] = R"(
cbuffer CropConstants : register(b0) {
    uint crop_x; uint crop_y; uint crop_width; uint crop_height;
    uint source_slice; uint pad0; uint pad1; uint pad2;
    float y_offset; float y_scale; float r_v; float g_u;
    float g_v; float b_u; float unused0; float unused1;
};
Texture2DArray<float> luma_plane : register(t0);
Texture2DArray<float2> chroma_plane : register(t1);
struct VertexOutput { float4 position : SV_Position; float2 uv : TEXCOORD0; };
VertexOutput vs_main(uint id : SV_VertexID) {
    const float2 positions[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
    const float2 uvs[3] = { float2(0.0, 1.0), float2(0.0, -1.0), float2(2.0, 1.0) };
    VertexOutput output; output.position = float4(positions[id], 0.0, 1.0); output.uv = uvs[id]; return output;
}
float4 ps_main(VertexOutput input) : SV_Target {
    uint2 local = min(uint2(input.uv * float2(crop_width, crop_height)), uint2(crop_width - 1, crop_height - 1));
    uint2 source = uint2(crop_x, crop_y) + local;
    float y = (luma_plane.Load(int4(source, source_slice, 0)) - y_offset) * y_scale;
    float2 uv = chroma_plane.Load(int4(source / 2, source_slice, 0)) - float2(0.5, 0.5);
    float r = saturate(y + r_v * uv.y);
    float g = saturate(y + g_u * uv.x + g_v * uv.y);
    float b = saturate(y + b_u * uv.x);
    return float4(r, g, b, 1.0);
}
)";

struct CropConstants {
    UINT crop_x, crop_y, crop_width, crop_height;
    UINT source_slice, pad0, pad1, pad2;
    float y_offset, y_scale, r_v, g_u, g_v, b_u, unused0, unused1;
};

class FfmpegD3D11Decoder {
public:
    ~FfmpegD3D11Decoder() { Close(); }

    int Open(const wchar_t* input_path, UINT x, UINT y, UINT width, UINT height, std::string* message) {
        const std::string path = Utf8FromWide(input_path);
        if (path.empty()) return Fail(AVERROR(EINVAL), "No video path was supplied.", message);
        int result = avformat_open_input(&format_, path.c_str(), nullptr, nullptr);
        if (result < 0) return Fail(result, "FFmpeg could not open this video: " + AvError(result), message);
        result = avformat_find_stream_info(format_, nullptr);
        if (result < 0) return Fail(result, "FFmpeg could not read stream information: " + AvError(result), message);
        stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index_ < 0) return Fail(stream_index_, "FFmpeg found no video stream.", message);
        const AVCodecParameters* parameters = format_->streams[stream_index_]->codecpar;
        if (x + width > static_cast<UINT>(parameters->width) || y + height > static_cast<UINT>(parameters->height) || width == 0 || height == 0) {
            return Fail(AVERROR(EINVAL), "Workspace crop exceeds the video dimensions.", message);
        }
        decoder_ = avcodec_find_decoder(parameters->codec_id);
        if (decoder_ == nullptr) return Fail(AVERROR_DECODER_NOT_FOUND, "FFmpeg has no decoder for this video codec.", message);
        codec_ = avcodec_alloc_context3(decoder_);
        if (codec_ == nullptr) return Fail(AVERROR(ENOMEM), "FFmpeg could not allocate a decoder context.", message);
        result = avcodec_parameters_to_context(codec_, parameters);
        if (result < 0) return Fail(result, "FFmpeg could not configure the decoder: " + AvError(result), message);
        result = av_hwdevice_ctx_create(&hardware_device_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (result < 0) return Fail(result, "FFmpeg D3D11VA device is unavailable: " + AvError(result), message);
        codec_->hw_device_ctx = av_buffer_ref(hardware_device_);
        // Tell FFmpeg to allocate decoder surfaces that can also be sampled by
        // the ROI shader. This avoids copying the entire decoded image into a
        // second NV12 texture every frame.
        AVBufferRef* hardware_frames = av_hwframe_ctx_alloc(hardware_device_);
        if (hardware_frames == nullptr) return Fail(AVERROR(ENOMEM), "FFmpeg could not allocate D3D11 frame context.", message);
        auto* frames = reinterpret_cast<AVHWFramesContext*>(hardware_frames->data);
        frames->format = AV_PIX_FMT_D3D11;
        frames->sw_format = AV_PIX_FMT_NV12;
        frames->width = parameters->width;
        frames->height = parameters->height;
        frames->initial_pool_size = 16;
        auto* d3d_frames = reinterpret_cast<AVD3D11VAFramesContext*>(frames->hwctx);
        d3d_frames->BindFlags = D3D11_BIND_DECODER | D3D11_BIND_SHADER_RESOURCE;
        result = av_hwframe_ctx_init(hardware_frames);
        if (result < 0) { av_buffer_unref(&hardware_frames); return Fail(result, "FFmpeg could not create shader-readable D3D11 decode surfaces: " + AvError(result), message); }
        codec_->hw_frames_ctx = hardware_frames;
        codec_->get_format = GetD3D11Format;
        result = avcodec_open2(codec_, decoder_, nullptr);
        if (result < 0) return Fail(result, "FFmpeg could not open a D3D11VA decoder: " + AvError(result), message);
        packet_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        if (packet_ == nullptr || frame_ == nullptr) return Fail(AVERROR(ENOMEM), "FFmpeg could not allocate frame storage.", message);
        AVHWDeviceContext* hardware = reinterpret_cast<AVHWDeviceContext*>(hardware_device_->data);
        auto* d3d = reinterpret_cast<AVD3D11VADeviceContext*>(hardware->hwctx);
        device_ = d3d->device;
        context_ = d3d->device_context;
        device_->AddRef();
        context_->AddRef();
        lock_ = d3d->lock;
        unlock_ = d3d->unlock;
        lock_context_ = d3d->lock_ctx;
        crop_ = {x, y, width, height, 0, 0, 0, 0, 16.0f / 255.0f, 255.0f / 219.0f, 1.596f, -0.392f, -0.813f, 2.017f, 0.0f, 0.0f};
        result = CreateResources(static_cast<UINT>(parameters->width), static_cast<UINT>(parameters->height));
        if (result < 0) return Fail(result, "Could not create D3D11 ROI resources.", message);
        *message = std::string("FFmpeg D3D11VA ready for ") + decoder_->name + " with GPU ROI crop.";
        return 0;
    }

    int ReadNext(unsigned char* destination, UINT capacity, int* frame_index, std::string* message) {
        if (destination == nullptr || frame_index == nullptr) return Fail(AVERROR(EINVAL), "Invalid output buffer.", message);
        const UINT required = crop_.crop_width * crop_.crop_height * 3;
        if (capacity < required) return Fail(AVERROR(ENOSPC), "Python ROI buffer is too small.", message);
        while (true) {
            int result = avcodec_receive_frame(codec_, frame_);
            if (result >= 0) {
                result = CopyFrameToBgr(destination, message);
                av_frame_unref(frame_);
                if (result < 0) return result;
                *frame_index = decoded_frame_index_++;
                return 0;
            }
            if (result != AVERROR(EAGAIN) && result != AVERROR_EOF) return Fail(result, "FFmpeg frame receive failed: " + AvError(result), message);
            while ((result = av_read_frame(format_, packet_)) >= 0) {
                if (packet_->stream_index == stream_index_) {
                    result = avcodec_send_packet(codec_, packet_);
                    av_packet_unref(packet_);
                    if (result < 0 && result != AVERROR(EAGAIN)) return Fail(result, "FFmpeg packet decode failed: " + AvError(result), message);
                    break;
                }
                av_packet_unref(packet_);
            }
            if (result < 0) {
                if (!flushed_) {
                    flushed_ = true;
                    avcodec_send_packet(codec_, nullptr);
                    continue;
                }
                return 1;
            }
        }
    }

    void Close() {
        output_staging_.Reset(); output_texture_.Reset(); output_rtv_.Reset();
        luma_srv_.Reset(); chroma_srv_.Reset(); source_texture_.Reset(); sampling_texture_.Reset();
        constant_buffer_.Reset(); vertex_shader_.Reset(); pixel_shader_.Reset();
        if (device_) device_->Release();
        if (context_) context_->Release();
        device_ = nullptr; context_ = nullptr;
        av_frame_free(&frame_); av_packet_free(&packet_); avcodec_free_context(&codec_);
        av_buffer_unref(&hardware_device_); avformat_close_input(&format_);
    }

private:
    int CreateResources(UINT source_width, UINT source_height) {
        (void)source_width; (void)source_height;
        D3D11_TEXTURE2D_DESC output{};
        output.Width = crop_.crop_width; output.Height = crop_.crop_height; output.MipLevels = 1; output.ArraySize = 1;
        output.Format = DXGI_FORMAT_B8G8R8A8_UNORM; output.SampleDesc.Count = 1; output.Usage = D3D11_USAGE_DEFAULT; output.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(device_->CreateTexture2D(&output, nullptr, &output_texture_))) return AVERROR_EXTERNAL;
        if (FAILED(device_->CreateRenderTargetView(output_texture_.Get(), nullptr, &output_rtv_))) return AVERROR_EXTERNAL;
        output.Usage = D3D11_USAGE_STAGING; output.BindFlags = 0; output.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(device_->CreateTexture2D(&output, nullptr, &output_staging_))) return AVERROR_EXTERNAL;
        D3D11_BUFFER_DESC constant{}; constant.ByteWidth = sizeof(CropConstants); constant.Usage = D3D11_USAGE_DYNAMIC; constant.BindFlags = D3D11_BIND_CONSTANT_BUFFER; constant.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device_->CreateBuffer(&constant, nullptr, &constant_buffer_))) return AVERROR_EXTERNAL;
        ComPtr<ID3DBlob> vertex_blob, pixel_blob, errors;
        if (FAILED(D3DCompile(kShaderSource, sizeof(kShaderSource) - 1, nullptr, nullptr, nullptr, "vs_main", "vs_5_0", 0, 0, &vertex_blob, &errors))) return AVERROR_EXTERNAL;
        if (FAILED(D3DCompile(kShaderSource, sizeof(kShaderSource) - 1, nullptr, nullptr, nullptr, "ps_main", "ps_5_0", 0, 0, &pixel_blob, &errors))) return AVERROR_EXTERNAL;
        if (FAILED(device_->CreateVertexShader(vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(), nullptr, &vertex_shader_))) return AVERROR_EXTERNAL;
        if (FAILED(device_->CreatePixelShader(pixel_blob->GetBufferPointer(), pixel_blob->GetBufferSize(), nullptr, &pixel_shader_))) return AVERROR_EXTERNAL;
        return 0;
    }

    int CopyFrameToBgr(unsigned char* destination, std::string* message) {
        if (frame_->format != AV_PIX_FMT_D3D11) return Fail(AVERROR(ENOSYS), "FFmpeg returned a CPU frame instead of D3D11VA.", message);
        auto* source = reinterpret_cast<ID3D11Texture2D*>(frame_->data[0]);
        const UINT source_slice = static_cast<UINT>(reinterpret_cast<intptr_t>(frame_->data[1]));
        if (source == nullptr) return Fail(AVERROR_EXTERNAL, "FFmpeg returned an empty D3D11 texture.", message);
        D3D11_TEXTURE2D_DESC description{}; source->GetDesc(&description);
        if (description.Format != DXGI_FORMAT_NV12) return Fail(AVERROR(ENOSYS), "Only NV12 hardware frames are currently supported.", message);
        if (lock_) lock_(lock_context_);
        HRESULT hr = EnsureSourceViews(source, source_slice);
        if (FAILED(hr)) {
            // A driver may reject shader-readable decode surfaces. Keep the
            // compatibility copy path for that driver, but profile it clearly.
            if (!sampling_texture_) {
                D3D11_TEXTURE2D_DESC copy_desc = description;
                copy_desc.ArraySize = 1; copy_desc.MipLevels = 1; copy_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                copy_desc.Usage = D3D11_USAGE_DEFAULT; copy_desc.CPUAccessFlags = 0; copy_desc.MiscFlags = 0;
                if (FAILED(device_->CreateTexture2D(&copy_desc, nullptr, &sampling_texture_))) { if (unlock_) unlock_(lock_context_); return Fail(AVERROR_EXTERNAL, "Could not create D3D11 compatibility sampling texture.", message); }
            }
            context_->CopySubresourceRegion(sampling_texture_.Get(), 0, 0, 0, 0, source, source_slice, nullptr);
            hr = CreateSourceViews(sampling_texture_.Get(), 0);
            if (FAILED(hr)) { if (unlock_) unlock_(lock_context_); return Fail(AVERROR_EXTERNAL, "Could not create D3D11 NV12 sampling views.", message); }
        }
        crop_.source_slice = source_texture_ ? source_slice : 0;
        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = context_->Map(constant_buffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (SUCCEEDED(hr)) {
            std::memcpy(mapped.pData, &crop_, sizeof(crop_));
            context_->Unmap(constant_buffer_.Get(), 0);
            ID3D11ShaderResourceView* views[] = {luma_srv_.Get(), chroma_srv_.Get()};
            ID3D11Buffer* constants[] = {constant_buffer_.Get()};
            context_->OMSetRenderTargets(1, output_rtv_.GetAddressOf(), nullptr);
            const D3D11_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(crop_.crop_width), static_cast<float>(crop_.crop_height), 0.0f, 1.0f};
            context_->RSSetViewports(1, &viewport);
            context_->IASetInputLayout(nullptr); context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context_->VSSetShader(vertex_shader_.Get(), nullptr, 0); context_->PSSetShader(pixel_shader_.Get(), nullptr, 0);
            context_->PSSetShaderResources(0, 2, views); context_->PSSetConstantBuffers(0, 1, constants);
            context_->Draw(3, 0);
            ID3D11ShaderResourceView* empty[] = {nullptr, nullptr}; context_->PSSetShaderResources(0, 2, empty);
            context_->CopyResource(output_staging_.Get(), output_texture_.Get());
            D3D11_MAPPED_SUBRESOURCE staging{};
            hr = context_->Map(output_staging_.Get(), 0, D3D11_MAP_READ, 0, &staging);
            if (SUCCEEDED(hr)) {
                for (UINT row = 0; row < crop_.crop_height; ++row) {
                    const auto* source_row = static_cast<const unsigned char*>(staging.pData) + row * staging.RowPitch;
                    auto* target_row = destination + row * crop_.crop_width * 3;
                    for (UINT column = 0; column < crop_.crop_width; ++column) {
                        target_row[column * 3 + 0] = source_row[column * 4 + 0];
                        target_row[column * 3 + 1] = source_row[column * 4 + 1];
                        target_row[column * 3 + 2] = source_row[column * 4 + 2];
                    }
                }
                context_->Unmap(output_staging_.Get(), 0);
            }
        }
        if (unlock_) unlock_(lock_context_);
        if (FAILED(hr)) return Fail(AVERROR_EXTERNAL, "D3D11 GPU crop/readback failed.", message);
        return 0;
    }

    HRESULT EnsureSourceViews(ID3D11Texture2D* source, UINT source_slice) {
        // Decoder-surface SRVs are driver-specific in practice. The Intel
        // driver accepted the view but produced a different plane layout than
        // the documented NV12 shader view, so retain the known-correct copy
        // into a shader-owned NV12 texture until that interop is validated.
        (void)source;
        (void)source_slice;
        return E_FAIL;
#if 0
        if (source_texture_.Get() == source && luma_srv_ && chroma_srv_) return S_OK;
        source_texture_.Reset(); luma_srv_.Reset(); chroma_srv_.Reset();
        source->AddRef(); source_texture_.Attach(source);
        return CreateSourceViews(source, source_slice);
#endif
    }

    HRESULT CreateSourceViews(ID3D11Texture2D* texture, UINT source_slice) {
        luma_srv_.Reset(); chroma_srv_.Reset();
        D3D11_TEXTURE2D_DESC description{}; texture->GetDesc(&description);
        D3D11_SHADER_RESOURCE_VIEW_DESC y_view{};
        y_view.Format = DXGI_FORMAT_R8_UNORM; y_view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
        y_view.Texture2DArray.MipLevels = 1; y_view.Texture2DArray.FirstArraySlice = 0; y_view.Texture2DArray.ArraySize = description.ArraySize;
        HRESULT hr = device_->CreateShaderResourceView(texture, &y_view, &luma_srv_);
        if (FAILED(hr)) return hr;
        D3D11_SHADER_RESOURCE_VIEW_DESC uv_view = y_view; uv_view.Format = DXGI_FORMAT_R8G8_UNORM;
        hr = device_->CreateShaderResourceView(texture, &uv_view, &chroma_srv_);
        if (FAILED(hr)) { luma_srv_.Reset(); return hr; }
        return S_OK;
    }

    int Fail(int code, const std::string& text, std::string* message) { if (message) *message = text; return code; }
    AVFormatContext* format_ = nullptr; AVCodecContext* codec_ = nullptr; const AVCodec* decoder_ = nullptr;
    AVBufferRef* hardware_device_ = nullptr; AVPacket* packet_ = nullptr; AVFrame* frame_ = nullptr; int stream_index_ = -1; int decoded_frame_index_ = 0; bool flushed_ = false;
    ID3D11Device* device_ = nullptr; ID3D11DeviceContext* context_ = nullptr; void (*lock_)(void*) = nullptr; void (*unlock_)(void*) = nullptr; void* lock_context_ = nullptr;
    ComPtr<ID3D11Texture2D> source_texture_, sampling_texture_, output_texture_, output_staging_; ComPtr<ID3D11RenderTargetView> output_rtv_; ComPtr<ID3D11ShaderResourceView> luma_srv_, chroma_srv_; ComPtr<ID3D11Buffer> constant_buffer_; ComPtr<ID3D11VertexShader> vertex_shader_; ComPtr<ID3D11PixelShader> pixel_shader_;
    CropConstants crop_{};
};

}  // namespace

extern "C" __declspec(dllexport) int ct_ffmpeg_d3d11_decoder_open(const wchar_t* path, unsigned int x, unsigned int y, unsigned int width, unsigned int height, void** handle, char* output, unsigned int capacity) {
    if (handle == nullptr) { WriteMessage(output, capacity, "No decoder handle output was supplied."); return AVERROR(EINVAL); }
    *handle = nullptr;
    auto* decoder = new FfmpegD3D11Decoder();
    std::string message;
    const int result = decoder->Open(path, x, y, width, height, &message);
    WriteMessage(output, capacity, message);
    if (result < 0) { delete decoder; return result; }
    *handle = decoder;
    return 0;
}

extern "C" __declspec(dllexport) int ct_ffmpeg_d3d11_decoder_next(void* handle, unsigned char* bgr, unsigned int capacity, int* frame_index, char* output, unsigned int message_capacity) {
    if (handle == nullptr) { WriteMessage(output, message_capacity, "Decoder handle is closed."); return AVERROR(EINVAL); }
    std::string message;
    const int result = static_cast<FfmpegD3D11Decoder*>(handle)->ReadNext(bgr, capacity, frame_index, &message);
    if (!message.empty()) WriteMessage(output, message_capacity, message);
    return result;
}

extern "C" __declspec(dllexport) void ct_ffmpeg_d3d11_decoder_close(void* handle) {
    delete static_cast<FfmpegD3D11Decoder*>(handle);
}
