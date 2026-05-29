Goal of this project
To compare and study how well YOLO recognizes silhouette shapes that are either constructed or natural shapes.

Data:
I have 5 categories of shapes. 
Source: A generative sampling procedure matches natural shape curvature statistics while removing
all global cues (Elder, Oleskiw, & Fruend, 2018). Local curvature is iteratively sampled while
preserving shape topology.

The original data is put as 2048x2048, transparent png images, with no background and white foreground.
Location: /Shapes
Minimum of 391 shapes per category.

From category 1 to 4, they are synthetic shapes. where category 5 is silhouette of animals.
Category 1: is unconstrained, with maximum entropy in curvature
Category 2: is local matched: variance
Category 3: is local matched: skew and/or kurtosis
Category 4: is local matched: all (varaince+kurtosis+skew)
Category 5: is natural animals


First experiment:
Generate x number of images, each image should have a grey background, and y - z range number of shapes in it with different scale, positioning, and rotation. The number of shapes present across all images should be equal in categories.

YOLO should be trained to correctly bound and classify each shape to its corrrect category.
Example: Shapes/cat1/img576.png, Shapes/cat1/img588.png should be labelled as cat1.

As a reference to this project: I have an assignment done in YOLO
Located in: /Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/assignment_3/Assignment3_pro.ipynb
