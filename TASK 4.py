# Install scikit-surprise before running this code using: pip install scikit-surprise

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Step 1: Prepare sample user-item ratings data
ratings_data = {
    "userId": [1, 1, 1, 2, 2, 3, 3, 3],
    "itemId": [1, 2, 3, 2, 3, 1, 2, 3],
    "rating": [5, 3, 2, 4, 1, 2, 5, 4],
}

ratings_df = pd.DataFrame(ratings_data)

# Step 2: Define a Reader to specify rating scale
rating_reader = Reader(rating_scale=(1, 5))

# Step 3: Load the dataset into surprise format
surprise_data = Dataset.load_from_df(ratings_df[["userId", "itemId", "rating"]], rating_reader)

# Step 4: Split the data into training and test sets
train_data, test_data = train_test_split(surprise_data, test_size=0.25, random_state=42)

# Step 5: Initialize and train the SVD algorithm
svd_model = SVD()
svd_model.fit(train_data)

# Step 6: Generate predictions on test data
predicted_ratings = svd_model.test(test_data)

# Step 7: Evaluate model performance using RMSE
rmse_score = accuracy.rmse(predicted_ratings)
print(f"Root Mean Squared Error (RMSE): {rmse_score:.4f}")

# Step 8: Make a prediction for a specific user and item
user_id = 1
item_id = 3
predicted = svd_model.predict(uid=user_id, iid=item_id)
print(f"Predicted rating for user {user_id} on item {item_id}: {predicted.est:.2f}")
