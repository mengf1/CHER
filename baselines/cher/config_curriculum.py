
learning_step = 0
total_learning_step = 200

learning_candidates = 128
learning_selected = 64 # batch_size
learning_up = 2
learning_down = 1
learning_rate = 0.0001
lambda_starter = 1
fixed_lambda = -1 # -1 starts \lambda-curriculum 
# FULL: FetchReach-v1, HandReach-v0, HandManipulateEggFull-v0 
# ROTATION: HandManipulateBlockRotateXYZ-v0, HandManipulatePenRotate-v0
goal_type = "FULL"
