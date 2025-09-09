# Food Recommendation System

This project is a hybrid Food Recommendation System designed to provide personalized food suggestions to users by leveraging both collaborative and content-based filtering techniques.

## Hybrid Recommendation Approach

Our system combines two powerful approaches to deliver more accurate and relevant recommendations:

### 1. Collaborative Filtering (KNN-based)

We use the K-Nearest Neighbors (KNN) algorithm to analyze user interactions and identify similar users (neighbors). By examining the preferences and behaviors of these nearest neighbors, the system can recommend dishes that users with similar tastes have enjoyed. This approach helps uncover popular dishes among like-minded individuals, improving the overall recommendation quality.

### 2. Content-Based Filtering (BERT-powered)

In addition to collaborative filtering, we employ a content-based method using BERT (Bidirectional Encoder Representations from Transformers). BERT enables the system to understand and compare the semantic content of dish descriptions. By analyzing the dishes a user has already liked, the model finds and recommends other dishes with similar characteristics.

## Why Docker?

A Dockerfile is included in this project to simplify deployment and ensure consistency across different environments. By containerizing the application:

- Developers and users can run the system without worrying about dependency conflicts.
- The environment is reproducible, making it easier to deploy the application on any machine or cloud provider.
- Docker streamlines testing and collaboration, as everyone works in the same environment.

## Summary

- **Hybrid Recommendation**: Combines collaborative (KNN) and content-based (BERT) methods for accurate suggestions.
- **Collaborative Filtering**: Finds similar users and recommends what they like.
- **Content-Based Filtering**: Uses BERT to compare dish descriptions and suggest similar items.
- **Docker Support**: Ensures easy setup, consistent deployment, and a reproducible environment.

Feel free to explore and contribute to the project!
