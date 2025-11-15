# This model needs to be trained on the data set every day and then using 
# the most recent prompt from the prompt.jsonl file needs to generate a post 
# I've kept all of the helper files with your specific logic 
# from before so you can use them to help you build the model. 

# The key bit I've changed is that we will manually do the metric updating in the 
# file And also the posting on Twitter because these are the difficult bits to automate

# Finally in the helpers folder, there's a quality control Python file which has some very 
# small almost pseudocode functions for controlling the quality of the models output 
# you might want to use these