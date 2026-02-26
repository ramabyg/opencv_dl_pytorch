#include <iostream>
#include <string>
#include <fstream>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace F = torch::nn::functional;
const int64_t kLogInterval = 10;


class Caltech : public torch::data::Dataset<Caltech>
{
    private:
        std::vector<std::string>image_paths;
        std::vector<int>labels;
        std::string data_path;
        torch::Tensor tensor_, target_;

        std::string join_paths(std::string head, const std::string& tail) 
        {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
        }

    public:
        explicit Caltech(const std::string& input_path, const std::string& output_path, const std::string& Path) 
     { 

            data_path = Path;
            // Read the image paths and store them
            std::ifstream file1(input_path);
            std::string curline1;
            while (std::getline(file1, curline1))
            {
                image_paths.push_back(curline1);
            }
            file1.close();  

            // Read the labels and store them
            std::ifstream file2(output_path);
            std::string curline2;
            while (std::getline(file2, curline2))
            {
                labels.push_back(std::stoi(curline2));
            }
            file2.close();  

    }

    /// Returns the length of the samples.
    torch::optional<size_t> size() const override
    { return image_paths.size(); }

    /// Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override
    {
        cv::Mat image = cv::imread(join_paths(data_path, image_paths[index]));
        cv::resize(image, image ,cv::Size(160, 160));

        torch::Tensor tensor_ = torch::from_blob(image.data, {image.rows, image.cols, 3}, at::kByte);
        tensor_ = tensor_.toType(at::kFloat);
        tensor_ = tensor_.permute({2, 0, 1});

        torch::Tensor target_ = torch::tensor(labels[index]);

        return { tensor_, target_ };
    }

 };

struct Net : torch::nn::Module
{
    Net(int64_t num_classes)    
    {
        //torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(p).stride(s) and similary other options
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3)));
        dp1 = register_module("dp1", torch::nn::Dropout(0.25));
        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
        dp2 = register_module("dp2", torch::nn::Dropout(0.25));
        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
        dp3 = register_module("dp3", torch::nn::Dropout(0.25));
        fc1 = register_module("fc1", torch::nn::Linear(2 * 2 * 64 * 81, 512));
        dp4 = register_module("dp4", torch::nn::Dropout(0.5));
        fc2 = register_module("fc2", torch::nn::Linear(512, num_classes));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(conv1_1->forward(x));
        x = torch::relu(conv1_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp1(x);

        x = torch::relu(conv2_1->forward(x));
        x = torch::relu(conv2_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp2(x);
        
        x = torch::relu(conv3_1->forward(x));
        x = torch::relu(conv3_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp3(x);

        x = x.view({-1, 2 * 2 * 64 * 81});
        
        x = torch::relu(fc1->forward(x));
        x = dp4(x);
        x = torch::log_softmax(fc2->forward(x), 1);
        
        return x;
    }

    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Dropout dp1{nullptr};
    torch::nn::Dropout dp2{nullptr};
    torch::nn::Dropout dp3{nullptr};
    torch::nn::Dropout dp4{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};

template <typename DataLoader>
void train(int32_t epoch, Net& model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size)
{
    model.train();
    double train_loss = 0;
    int32_t correct = 0;
    size_t batch_idx = 0;   
    for (auto& batch : data_loader) {
        auto x = batch.data.to(device), targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model.forward(x);
        auto loss = F::cross_entropy(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();
        
        train_loss += loss.template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
        batch_idx+=1;       
    }
    train_loss /= batch_idx;
    std::printf(
        "   Train set: Average loss: %.4f | Accuracy: %.3f",
        train_loss,
        static_cast<double>(correct) / dataset_size);
}

template <typename DataLoader>
void test(Net& model, torch::Device device, DataLoader& data_loader, size_t dataset_size)
{
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto output = model.forward(data);
        test_loss += F::cross_entropy(
            output,
            targets,
            F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kSum))
            .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::printf(
        "\n   Test set: Average loss: %.4f | Accuracy: %.3f\n",
        test_loss,
        static_cast<double>(correct) / dataset_size);
}


int main()

{
    int numEpochs = 8;
    int trainBatchSize = 32;
    int testBatchSize = 16;

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // Hybrid CaltechClassifer(4);
    Net CaltechClassifier(4);
    CaltechClassifier.to(device);

    auto trainData = Caltech("./caltech_subset/train_paths.txt","./caltech_subset/train_labels.txt", "./caltech_subset/train").map(torch::data::transforms::Stack<>());
    auto testData = Caltech("./caltech_subset/test_paths.txt","./caltech_subset/test_labels.txt", "./caltech_subset/test").map(torch::data::transforms::Stack<>());

    auto trainDataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(trainData, trainBatchSize);
    auto testDataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(testData, testBatchSize);



    const int64_t trainLen = trainData.size().value();
    const int64_t testLen = testData.size().value();


    torch::optim::Adam optimizer(CaltechClassifier.parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5,0.9)));


    for (size_t epoch = 1; epoch <= numEpochs; ++epoch) 
    {   std::cout<<"\nEpoch "<<epoch<<" statistics."<<std::endl;
        train(epoch, CaltechClassifier, device, *trainDataloader, optimizer, trainLen);
        test(CaltechClassifier, device, *testDataloader, testLen);
    }





    return 0;
}
