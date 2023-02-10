# Predicting-Dog-Age-from-Teeth

Abstract

Estimating the age of stray dogs is a challenge due to their unknown history. However, determining a dog's age is crucial for providing the best care, particularly for choosing medical treatments and conducting disease tests. Currently, the most common method for estimating a dog's age is based on physical appearance, which is subjective and unreliable, leading to incorrect assumptions about a dog's age, with serious consequences such as incorrect dosages of medication. Fortunately, recent studies show that dental development is a reliable indicator of age in dogs, as their teeth erupt and wear at a predictable rate. To address this issue, I propose an app that uses a multi-input model to estimate a dog’s age based on an image of its teeth and information about its breed and dental health. I will fine-tune the ResNet50 model on images collected from veterinary clinics and train a neural network on the dog’s breed (small or large mix)  and the dog's dental condition (whether the dog receives regular cleaning or not). Both of the model’s outputs will serve as the input features for the final fully connected layers which will output an age group of 2 in the range 0 to 15. Having two separate models for both input types allows the model to learn the dependencies between the image and the dog’s breed and dental condition. The app’s goal is to provide animal shelter workers with an affordable method for dog age estimation that improves upon current time-consuming and subjective manual methods.


Proposal: https://docs.google.com/document/d/1pr-bKySfzpsYCoMGMbm_E9jx-c4QZWQBa9z-q2wi6sk/edit#

For any questions/comments email: 2redrosen@gmail.com
