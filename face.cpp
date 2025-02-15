#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <torch/torch.h>

class FaceDetector {
public:
    FaceDetector(const cv::CascadeClassifier &face_cascade)
        : face_cascade(face_cascade) {}

    void detect(cv::Mat& img) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);  // Enhance contrast

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        // If no face detected, show "Looking to a bitch" in the middle of the screen with transparency
        if (faces.empty()) {
            displayTransparentText(img, "Looking for a bitch");
        }

        // For each face detected, draw rectangle and show message
        for (const auto& face : faces) {
            cv::rectangle(img, face, cv::Scalar(0, 0, 255), 2); // Blue rectangle for faces
            // Show "BITCH DETECTED" on top of the face
            cv::putText(img, "BITCH DETECTED", cv::Point(face.x, face.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }

private:
    cv::CascadeClassifier face_cascade;

    // Function to display transparent text in the center of the screen
    void displayTransparentText(cv::Mat& img, const std::string& text) {
        // Create a transparent overlay
        cv::Mat overlay = img.clone();
        overlay.setTo(cv::Scalar(0, 0, 0)); // Set a black overlay

        // Get the text size
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        cv::Point textOrg((img.cols - textSize.width) / 2, (img.rows + textSize.height) / 2);

        // Draw the text onto the overlay with transparency
        cv::putText(overlay, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

        // Blend the overlay with the original image
        double alpha = 0.8;  // Transparency factor
        cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);
    }
};

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::filesystem::path cascades_folder = "data/cascades";

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load((cascades_folder / "haarcascade_frontalface_default.xml").string())) {
        std::cerr << "Error loading face cascade" << std::endl;
        return 1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to access the webcam" << std::endl;
        return 1;
    }

    FaceDetector detector(face_cascade);
    cv::namedWindow("Face Detection", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Warning: Empty frame captured" << std::endl;
            continue;
        }

        detector.detect(frame);

        cv::imshow("Face Detection", frame);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Exiting face detection. Goodbye!" << std::endl;

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    return 0;
}
