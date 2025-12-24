•	Section 1: project overview
o	Briefly describe the BBO capstone project and its purpose
-The capstone project is a function maxima optimization project which spans multiple weeks and multi dimensional functions. 
o	What is the overall goal of the BBO capstone project? Why is it relevant in real-world ML? What’s the high-level idea?
-The overall goal is to use machine learning models of increasing complexity to optimize the input variable queries to achieve the maxima discrete output from the function. As more data become available weekly the models should be able to learn more and improve in accuracy of inputs per weekly variable. This is relevant across many industries and teaches working with unknown data sets alongside model optimization
o	How would this BBO capstone project support you in your current or future career?
-The overall goal teaches many different aspects that are currently applicable in my career: project timeline/planning, baseline modeling, 

•	Section 2: inputs and outputs
o	Clearly state what your model receives and returns.
-The model receives input features/variables for each function and returns continuous number/Y hats. Each function has a different number of input variables but receives the same amount(one) output per week
o	What are the inputs (query format, dimensions, constraints, etc.)? What is the expected output (response value, performance signal, etc.)? Include example formats, if possible.
Each weekly submission must be a positive number which lets us know this is a maximization problem. One example is function one which takes 2 input numbers: one example weekly input for function 1, a 2d function would be 0.40000-0.600000. The performance is shown by increasing the output variable each week vs past weeks.

•	Section 3: challenge objectives

o	Outline what you are trying to achieve within the BBO capstone project
I am trying to achieve the highest response value per function using multiple machine learning models to derive my input queries weekly.
o	Is the goal to minimise or maximise the function(s)? What constraints or limitations must you consider (e.g. number of queries, response delay and unknown function structure)?
The project is a maximization optimization goal. One constraint is that each number must be a positive number with a 6 digit format. Each week every function allows a set amount of input variables depending upon the dimensionality of the function but each function only returns one output response at the end of the week

•	Section 4: technical approach
o	Describe the strategies you used across your first three query submissions. You’re encouraged to treat this section as a living record – continue updating it as your approach evolves throughout the BBO capstone project.
-The first 2 weeks I used fairly simple statistical guessing techniques given to prior data points to model. On week 3 I used Gaussian process model to select my query input variables. I made a mistake though and trained the model on all function inputs/outputs when I should’ve built a model on each function. This was due to lack of time and has now been corrected
o	What ML methods or heuristics do you use? Will you model the unknown function? Would you consider using SVMs, regressions or Bayesian techniques? 
I am currently using: regression, random forest, gradient boosting, GP/bayes optimization and neural networks
o	How do you balance exploration and exploitation? What makes your approach thoughtful or unique?
I am currently running 5 models for each function per week to see the varying submissions choices and expected output performance increase per model. So far I have chosen to use mostly GP models despite the other model’s expectation of higher performance/greater maximization. As I have more data I will start to use multiple for advanced modeling technique submissions per model weekly to have a higher degree of exploration. I not only keep all of my work on github I also keep each function and weekly progress in a gemini 3.0 chat alongside claude opus 4.5 chat for feedback.

