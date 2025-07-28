
# import libs
import numpy as np 
from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split



# import dataset
data = Dataset.load_builtin("ml-100k") # user id, film id and rating


# train test split
trainset, testset = train_test_split(data, test_size = 0.2)



# define KNN model
model_options = {
    "name": "cosine",
    "user-based": True # similarity user-based
    }


# train operation
model = KNNBasic(sim_options= model_options)
model.fit(trainset)



# test operation
prediction = model.test(testset)
accuracy.rmse(prediction)



# recommendation system 
def get_top_n(predictions, n = 10):
    top_n = {}

    for uid, iid, true_r, est, _ in predictions:
        if not top_n.get(uid):
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse = True)
        top_n[uid] = user_ratings[:n]
        
    return top_n


n = 5
top_n = get_top_n(prediction, 5)

user_id = "2"
print(f"top: {n} recommendation for user {user_id}")

for item_id, rating in top_n[user_id]:
    print(f"item id: {item_id}, score: {rating}")






















