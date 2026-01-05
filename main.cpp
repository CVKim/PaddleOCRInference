#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <map>
#include <random>
#include <iomanip>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// TensorRT & CUDA
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// =================================================================================
// 1. Logger & Utils
// =================================================================================
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

std::vector<std::string> LoadDictionary(const std::string& path) {
    std::vector<std::string> chars;
    chars.push_back("blank");
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Error] Dictionary file not found: " << path << std::endl;
        return chars;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        chars.push_back(line);
    }
    return chars;
}


void OrderPoints(std::vector<cv::Point>& pts) {
    if (pts.size() != 4) return;

    std::vector<cv::Point> result(4);
    std::vector<int> sum_vals, diff_vals;

    for (const auto& p : pts) {
        sum_vals.push_back(p.x + p.y);
        diff_vals.push_back(p.y - p.x);
    }

    auto min_sum = std::min_element(sum_vals.begin(), sum_vals.end());
    auto max_sum = std::max_element(sum_vals.begin(), sum_vals.end());
    result[0] = pts[std::distance(sum_vals.begin(), min_sum)]; // TL
    result[2] = pts[std::distance(sum_vals.begin(), max_sum)]; // BR

    auto min_diff = std::min_element(diff_vals.begin(), diff_vals.end());
    auto max_diff = std::max_element(diff_vals.begin(), diff_vals.end());
    result[1] = pts[std::distance(diff_vals.begin(), min_diff)]; // TR
    result[3] = pts[std::distance(diff_vals.begin(), max_diff)]; // BL

    pts = result;
}

// =================================================================================
// 2. Visualization
// =================================================================================
cv::Mat VisualizeResult(const cv::Mat& src,
    const std::vector<std::vector<cv::Point>>& boxes,
    const std::vector<std::string>& texts,
    const std::vector<float>& rec_scores,
    const std::vector<std::string>& angles,
    const std::vector<float>& ang_scores,
    float drop_score = 0.5f) {

    cv::Mat imgLeft = src.clone();
    cv::Mat overlay = src.clone();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (i < rec_scores.size() && rec_scores[i] < drop_score) continue;

        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{boxes[i]}, color);
    }
    cv::addWeighted(overlay, 0.5, imgLeft, 0.5, 0, imgLeft);

    cv::Mat imgRight(src.rows, src.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (i >= texts.size()) break;
        
        if (rec_scores[i] < drop_score) continue;

        cv::Rect rect = cv::boundingRect(boxes[i]);
        int min_x = std::max(0, rect.x);
        int min_y = std::max(0, rect.y);

        std::stringstream ss_rec, ss_conf, ss_ang;
        ss_rec << "Rec text: " << texts[i];
        ss_conf << "Rec conf: " << std::fixed << std::setprecision(2) << rec_scores[i];

        std::string ang_str = (i < angles.size()) ? angles[i] : "0";
        float ang_conf = (i < ang_scores.size()) ? ang_scores[i] : 0.0f;
        ss_ang << "Ang: " << ang_str << ", Ang conf: " << std::fixed << std::setprecision(2) << ang_conf;

        double fontScale = (src.rows > 1000) ? 0.8 : 0.4;
        int thickness = 1;
        int lineStep = (int)(20 * (fontScale / 0.4));

        cv::Scalar textColor(0, 0, 0);

        cv::putText(imgRight, ss_rec.str(), cv::Point(min_x, min_y), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
        cv::putText(imgRight, ss_conf.str(), cv::Point(min_x, min_y + lineStep), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
        cv::putText(imgRight, ss_ang.str(), cv::Point(min_x, min_y + lineStep * 2), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
    }

    cv::Mat result;
    cv::hconcat(imgLeft, imgRight, result);
    return result;
}

// =================================================================================
// 3. TensorRT Engine Wrapper
// =================================================================================
class TensorRTEngine {
public:
    TensorRTEngine(const std::string& enginePath) { LoadEngine(enginePath); }
    ~TensorRTEngine() {
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
        for (auto& pair : buffers) cudaFree(pair.second);
    }

    void LoadEngine(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            std::cerr << "[Error] Engine file read error: " << path << std::endl;
            return;
        }
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(buffer.data(), size);
        context = engine->createExecutionContext();

        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) inputNames.push_back(name);
            else outputNames.push_back(name);
        }
    }

    void Infer(const std::vector<float>& inputHost, std::vector<float>& outputHost,
        int batchSize, int c, int h, int w, int& outH, int& outW) {

        if (!inputNames.empty()) context->setInputShape(inputNames[0].c_str(), Dims4{ batchSize, c, h, w });

        size_t inputByteSize = inputHost.size() * sizeof(float);
        if (bufferSizes[inputNames[0]] < inputByteSize) {
            if (buffers[inputNames[0]]) cudaFree(buffers[inputNames[0]]);
            cudaMalloc(&buffers[inputNames[0]], inputByteSize * 1.5);
            bufferSizes[inputNames[0]] = inputByteSize * 1.5;
        }
        cudaMemcpyAsync(buffers[inputNames[0]], inputHost.data(), inputByteSize, cudaMemcpyHostToDevice, stream);

        for (const auto& outName : outputNames) {
            Dims dims = context->getTensorShape(outName.c_str());
            size_t outElements = batchSize;
            for (int i = 1; i < dims.nbDims; ++i) outElements *= (dims.d[i] < 0 ? 1 : dims.d[i]);
            if (outElements <= (size_t)batchSize) outElements = batchSize * c * h * w;

            size_t byteSize = outElements * sizeof(float);
            if (bufferSizes[outName] < byteSize) {
                if (buffers[outName]) cudaFree(buffers[outName]);
                cudaMalloc(&buffers[outName], byteSize * 1.5);
                bufferSizes[outName] = byteSize * 1.5;
            }
            context->setTensorAddress(outName.c_str(), buffers[outName]);
        }
        context->setInputTensorAddress(inputNames[0].c_str(), buffers[inputNames[0]]);

        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        std::string targetOutName = outputNames[0];
        if (outputNames.size() > 1) {
            float bestMaxVal = -999.0f;
            for (const auto& outName : outputNames) {
                Dims dims = context->getTensorShape(outName.c_str());
                size_t outElements = 1;
                for (int i = 0; i < dims.nbDims; ++i) outElements *= (dims.d[i] < 0 ? 1 : dims.d[i]);
                std::vector<float> tempHost(outElements);
                cudaMemcpy(tempHost.data(), buffers[outName], outElements * sizeof(float), cudaMemcpyDeviceToHost);
                float maxVal = -999.0f;
                for (float v : tempHost) if (v > maxVal) maxVal = v;
                if (maxVal >= 0.0f && maxVal <= 1.0f && maxVal > bestMaxVal) {
                    bestMaxVal = maxVal; targetOutName = outName;
                }
            }
        }

        Dims finalDims = context->getTensorShape(targetOutName.c_str());
        size_t finalElements = 1;
        for (int i = 0; i < finalDims.nbDims; ++i) finalElements *= (finalDims.d[i] < 0 ? 1 : finalDims.d[i]);

        outputHost.resize(finalElements);
        cudaMemcpyAsync(outputHost.data(), buffers[targetOutName], finalElements * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (finalDims.nbDims == 4) { outH = finalDims.d[2]; outW = finalDims.d[3]; }
        else if (finalDims.nbDims == 3) { outH = finalDims.d[1]; outW = finalDims.d[2]; }
        else { outH = 1; outW = finalElements; }
    }

private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::map<std::string, void*> buffers;
    std::map<std::string, size_t> bufferSizes;
};

// =================================================================================
// 4. Preprocessing Functions (Correct Padding & Normalization)
// =================================================================================

// [Det] Dynamic, BGR Input, ImageNet Mean/Std
void PreprocessDet_Dynamic(const cv::Mat& img, std::vector<float>& output,
    int& out_h, int& out_w, float& ratio_h, float& ratio_w,
    const std::string& debugPath = "") {
    int max_side_len = 960;
    int h = img.rows; int w = img.cols;
    float ratio = 1.0f;
    if (std::max(h, w) > max_side_len) ratio = (h > w) ? (float)max_side_len / h : (float)max_side_len / w;

    int resize_h = int(h * ratio);
    int resize_w = int(w * ratio);
    resize_h = std::max((int)(round(resize_h / 32.0) * 32), 32);
    resize_w = std::max((int)(round(resize_w / 32.0) * 32), 32);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));

    if (!debugPath.empty()) cv::imwrite(debugPath, resized);

    ratio_h = (float)resize_h / h;
    ratio_w = (float)resize_w / w;
    out_h = resize_h;
    out_w = resize_w;

    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    output.resize(3 * resize_h * resize_w);
    float* out_ptr = output.data();

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    // Detection uses ImageNet Mean/Std
    float mean[] = { 0.485f, 0.456f, 0.406f };
    float std[] = { 0.229f, 0.224f, 0.225f };

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std[c];
        std::memcpy(out_ptr + c * resize_h * resize_w, channels[c].data, resize_h * resize_w * sizeof(float));
    }
}

// [Cls] Padding Value Fix (0.0f instead of -1.0f)
void PreprocessCls(const cv::Mat& img, std::vector<float>& output, const std::string& debugPath = "") {
    int h = 80, w = 160;
    float ratio = std::min((float)h / img.rows, (float)w / img.cols);
    int new_h = (int)(img.rows * ratio);
    int new_w = (int)(img.cols * ratio);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // Debug save (Before normalization, visual check)
    if (!debugPath.empty()) {
        cv::Mat debugImg = cv::Mat::zeros(h, w, CV_8UC3);
        resized.copyTo(debugImg(cv::Rect(0, 0, new_w, new_h)));
        cv::imwrite(debugPath, debugImg);
    }

    // 1. Initialize output with 0.0f (padding value)
    output.assign(3 * h * w, 0.0f);

    // 2. Normalize ROI only
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    float mean = 0.5f;
    float std = 0.5f;
    float* out_ptr = output.data();

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < new_h; ++i) { // Only up to resized height
            for (int j = 0; j < new_w; ++j) { // Only up to resized width
                float pixel = float_img.at<cv::Vec3f>(i, j)[c];
                out_ptr[c * h * w + i * w + j] = (pixel - mean) / std;
            }
        }
    }
}

// [Rec] Padding Value Fix (0.0f instead of -1.0f)
void PreprocessRec_Static(const cv::Mat& img, std::vector<float>& output, const std::string& debugPath = "") {
    int h_target = 48, w_target = 320;
    float ratio = (float)img.cols / (float)img.rows;
    int resize_w = (int)(h_target * ratio);
    if (resize_w > w_target) resize_w = w_target;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, h_target));

    if (!debugPath.empty()) {
        cv::Mat debugImg = cv::Mat::zeros(h_target, w_target, CV_8UC3);
        resized.copyTo(debugImg(cv::Rect(0, 0, resize_w, h_target)));
        cv::imwrite(debugPath, debugImg);
    }

    // 1. Initialize output with 0.0f
    output.assign(3 * h_target * w_target, 0.0f);

    // 2. Normalize ROI only
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    float mean = 0.5f;
    float std = 0.5f;
    float* out_ptr = output.data();

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < h_target; ++i) {
            for (int j = 0; j < resize_w; ++j) { // Only up to resize_w
                float pixel = float_img.at<cv::Vec3f>(i, j)[c];
                out_ptr[c * h_target * w_target + i * w_target + j] = (pixel - mean) / std;
            }
        }
    }
}

// [PostProcess]
std::vector<std::vector<cv::Point>> PostprocessDet(const std::vector<float>& pred, int map_h, int map_w, float thresh, float r_h, float r_w) {
    cv::Mat map(map_h, map_w, CV_32F, (void*)pred.data());
    cv::Mat binary;
    cv::threshold(map, binary, thresh, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> boxes;

    for (const auto& contour : contours) {
        cv::RotatedRect box = cv::minAreaRect(contour);
        if (std::min(box.size.width, box.size.height) < 3) continue;

        float area = cv::contourArea(contour);
        float perimeter = cv::arcLength(contour, true);
        float unclip_ratio = 1.5f;
        float distance = area * unclip_ratio / perimeter;
        cv::Size2f newSize(box.size.width + distance * 2, box.size.height + distance * 2);
        box.size = newSize;
        cv::Point2f vtx[4];
        box.points(vtx);
        std::vector<cv::Point> int_box;
        for (int i = 0; i < 4; i++) {
            float x = vtx[i].x / r_w;
            float y = vtx[i].y / r_h;
            int_box.push_back(cv::Point((int)x, (int)y));
        }
        OrderPoints(int_box);
        boxes.push_back(int_box);
    }
    return boxes;
}

cv::Mat GetRotateCropImage(const cv::Mat& img, std::vector<cv::Point> box, float margin_ratio = 1.2f) {
    cv::Point2f pts[4];
    for (int i = 0; i < 4; i++) pts[i] = cv::Point2f((float)box[i].x, (float)box[i].y);

    float width = std::max(norm(pts[0] - pts[1]), norm(pts[2] - pts[3]));
    float height = std::max(norm(pts[0] - pts[3]), norm(pts[1] - pts[2]));

    float margin_w = width * (margin_ratio - 1.0f) * 0.5f;
    float margin_h = height * (margin_ratio - 1.0f) * 0.5f;

    int target_w = (int)(width + margin_w * 2);
    int target_h = (int)(height + margin_h * 2);

    cv::Point2f dstPts[4] = {
        {margin_w, margin_h},
        {width + margin_w, margin_h},
        {width + margin_w, height + margin_h},
        {margin_w, height + margin_h}
    };

    cv::Mat M = cv::getPerspectiveTransform(pts, dstPts);
    cv::Mat dst;

    // 이미지를 사이즈를 벗어나도 에러가 나지 않고, 깔끔한 검은 테두리가 생깁니다.
    cv::warpPerspective(img, dst, M, cv::Size(target_w, target_h),
        cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    if ((float)dst.rows / (float)dst.cols >= 1.5) {
        cv::rotate(dst, dst, cv::ROTATE_90_CLOCKWISE);
    }
    return dst;
}

// =================================================================================
// 6. Main
// =================================================================================
int main() {
    std::string modelDir = "E:\\ONNX_TO_TRT\\";
    std::string detPath = modelDir + "det_model_dynamic.trt";
    std::string clsPath = modelDir + "cls_model.trt";
    std::string recPath = modelDir + "rec_model.trt"; // Static 320

    std::string dictPath = R"(E:\DL_SW\athenapaddleocr\athenapaddleocr\ppocr\utils\dict\ppocrv5_dict.txt)";
    std::string imgDir = "E:\\DL_SW\\athenapaddleocr\\msb_combined_data";
    std::string outDir = "./output_cpp";

    fs::create_directories(outDir + "/pre-proc");

    auto charDict = LoadDictionary(dictPath);
    std::cout << "[Info] Dictionary Loaded: " << charDict.size() << std::endl;

    TensorRTEngine detEngine(detPath);
    TensorRTEngine clsEngine(clsPath);
    TensorRTEngine recEngine(recPath);
    std::filesystem::create_directory(outDir);

    std::vector<std::string> images;
    for (const auto& entry : std::filesystem::directory_iterator(imgDir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".png" || ext == ".bmp") images.push_back(entry.path().string());
    }

    for (const auto& imgPath : images) {
        std::cout << "Processing: " << imgPath << std::endl;
        cv::Mat frame = cv::imread(imgPath);
        if (frame.empty()) continue;


        // test
        cv::Mat lab_image;
        cv::cvtColor(frame, lab_image, cv::COLOR_BGR2Lab); // LAB 색공간 변환

        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_image, lab_planes);

        auto clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
        clahe->apply(lab_planes[0], lab_planes[0]);

        cv::merge(lab_planes, lab_image);
        cv::cvtColor(lab_image, frame, cv::COLOR_Lab2BGR); // 다시 BGR로 변환

        std::string fileNameOnly = fs::path(imgPath).filename().string();

        // 1. Detection (BGR)
        std::vector<float> detInput, detOutput;
        int infer_h, infer_w;
        float r_h, r_w;

        std::string detDebugPath = outDir + "/pre-proc/det_pre_" + fileNameOnly;
        PreprocessDet_Dynamic(frame, detInput, infer_h, infer_w, r_h, r_w, detDebugPath);

        int outH, outW;
        detEngine.Infer(detInput, detOutput, 1, 3, infer_h, infer_w, outH, outW);
        auto boxes = PostprocessDet(detOutput, outH, outW, 0.3f, r_h, r_w);

        std::cout << "  -> Found " << boxes.size() << " boxes." << std::endl;

        std::vector<std::string> texts;
        std::vector<float> rec_scores;
        std::vector<std::string> ang_labels;
        std::vector<float> ang_scores;

        int box_idx = 0;
        for (const auto& box : boxes) {
            // Rec/Cls 전처리에 BGR Crop 그대로 사용
            cv::Mat crop = GetRotateCropImage(frame, box, 1.1f);

            // Cls
            std::string clsDebugPath = outDir + "/pre-proc/cls_pre_" + fileNameOnly + "_" + to_string(box_idx) + ".bmp";
            std::vector<float> clsInput, clsOutput;
            PreprocessCls(crop, clsInput, clsDebugPath);
            int dH, dW;
            clsEngine.Infer(clsInput, clsOutput, 1, 3, 80, 160, dH, dW);

            std::string ang_label = "0";
            float ang_score = clsOutput[0];
            if (clsOutput[1] > clsOutput[0]) {
                ang_label = "180";
                ang_score = clsOutput[1];
                if (ang_score > 0.9) cv::rotate(crop, crop, cv::ROTATE_180);
            }
            ang_labels.push_back(ang_label);
            ang_scores.push_back(ang_score);

            // Rec
            std::string recDebugPath = outDir + "/pre-proc/rec_pre_" + fileNameOnly + "_" + to_string(box_idx) + ".bmp";
            std::vector<float> recInput, recOutput;
            PreprocessRec_Static(crop, recInput, recDebugPath);

            int recTime, recVocab;
            recEngine.Infer(recInput, recOutput, 1, 3, 48, 320, recTime, recVocab);

            std::string text = "";
            float scoreSum = 0;
            int validCharCount = 0;
            int lastIndex = -1;

            for (int t = 0; t < recTime; t++) {
                int maxIdx = -1;
                float maxVal = -1e9;
                for (int v = 0; v < recVocab; v++) {
                    float val = recOutput[t * recVocab + v];
                    if (val > maxVal) { maxVal = val; maxIdx = v; }
                }
                if (maxIdx != lastIndex && maxIdx != 0) {
                    if (maxIdx < charDict.size()) {
                        text += charDict[maxIdx];
                        scoreSum += maxVal;
                        validCharCount++;
                    }
                }
                lastIndex = maxIdx;
            }
            float meanScore = validCharCount > 0 ? scoreSum / validCharCount : 0.0f;
            texts.push_back(text);
            rec_scores.push_back(meanScore);

            std::cout << "    Rec Text : " << text << std::endl;
            std::cout << "    Rec Conf : " << meanScore << std::endl;
            std::cout << "    Ang : " << ang_label << ", Ang Conf : " << ang_score << std::endl;

            box_idx++;
        }
        cv::Mat resultImg = VisualizeResult(frame, boxes, texts, rec_scores, ang_labels, ang_scores, 0.0f);

        std::string fileName = std::filesystem::path(imgPath).filename().string();
        cv::imwrite(outDir + "/" + fileName, resultImg);
    }
    std::cout << "All Finished." << std::endl;
    return 0;
}