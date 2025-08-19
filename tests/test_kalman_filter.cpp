#include <gtest/gtest.h>
#include "kalman_filter.h"
#include <Eigen/Dense>

using namespace tracking;

class KalmanFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        filter = std::make_unique<ExtendedKalmanFilter>();
    }

    std::unique_ptr<ExtendedKalmanFilter> filter;
};

TEST_F(KalmanFilterTest, InitializationTest) {
    EXPECT_FALSE(filter->isInitialized());
    
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(9);
    initial_state << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    filter->initialize(initial_state);
    EXPECT_TRUE(filter->isInitialized());
    
    Eigen::VectorXd state = filter->getState();
    EXPECT_NEAR(state(0), 1.0, 1e-6);
    EXPECT_NEAR(state(1), 2.0, 1e-6);
    EXPECT_NEAR(state(2), 3.0, 1e-6);
}

TEST_F(KalmanFilterTest, PredictionTest) {
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(9);
    initial_state << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    
    filter->initialize(initial_state);
    
    // Predict for 1 second
    filter->predict(1.0);
    
    Eigen::VectorXd predicted_state = filter->getState();
    
    // With constant velocity, position should change by velocity * time
    EXPECT_NEAR(predicted_state(0), 1.0, 1e-6); // x position
    EXPECT_NEAR(predicted_state(1), 1.0, 1e-6); // y position
    EXPECT_NEAR(predicted_state(2), 0.0, 1e-6); // z position
}

TEST_F(KalmanFilterTest, LiDARUpdateTest) {
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(9);
    initial_state << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    filter->initialize(initial_state);
    
    // Create LiDAR measurement
    Eigen::VectorXd measurement(3);
    measurement << 1.1, 2.1, 3.1;
    
    Eigen::MatrixXd R_lidar = Eigen::MatrixXd::Identity(3, 3) * 0.1;
    
    filter->updateLiDAR(measurement, R_lidar);
    
    Eigen::VectorXd updated_state = filter->getState();
    
    // State should be updated towards measurement
    EXPECT_GT(updated_state(0), 1.0);
    EXPECT_GT(updated_state(1), 2.0);
    EXPECT_GT(updated_state(2), 3.0);
}

TEST_F(KalmanFilterTest, CovarianceTest) {
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(9);
    filter->initialize(initial_state);
    
    Eigen::MatrixXd initial_cov = filter->getCovariance();
    EXPECT_EQ(initial_cov.rows(), 9);
    EXPECT_EQ(initial_cov.cols(), 9);
    
    // Covariance should be positive definite
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(initial_cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    
    for (int i = 0; i < eigenvalues.size(); ++i) {
        EXPECT_GT(eigenvalues(i), 0.0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
