####################################
# Week 05: Part 02
####################################


# Exercises: DAOST chapter 3

# Looking at Fig 3-1, Janert writes
# "the data itself shows clearly that the amount of random noise in the data is small". 
# What do you think his argument is?
"""
Looking at the data alone it's obvious that they are not randomly distributed. 
However, there is a clear complicated a relationship between x and y. 
This doesn't mean that there aren't noise with the data but only that any noise will be systematic so it cancel out.
"""

# Can you think of a real-world example of a multivariate relationship like the one in Fig 3-3 (lower right panel)?
"""
Not sure about this question but here is a possible example: 
[ref: https://math.stackexchange.com/questions/1097745/example-of-a-real-world-situation-where-multivariate-analysis-is-applicable]
"""

# What are the two methods Janert metions for smoothing noisy data? Can you think of other ones?
"""
splines and LOESS 
other methodes could be: Moving Average and Simple Exponential 
[ref: https://corporatefinanceinstitute.com/resources/knowledge/other/data-smoothing/]
"""

# What is problematic about using straight lines to fit the data in Fig 3-5? 
# Something similar is actually the topic of a Nature article from 2004 get it here:
# https://github.com/suneman/socialdataanalysis2018/blob/master/files/regrunners.pdf 
"""
Using a straight-line to fit the data prior to 1990 shows that 
women should start finishing fatster than men soon after 1990 and continue to do so at high rate. 
but this is not the case according to real data after 1990 where women finishing times leveled off! 
While using a LOESS smoothed curves to the same data prior to 1990 shows that 
women finishing times begun to level off before the year 1990.

This is due to the fact that smoothed curves have can detected local features and follow the data on smaller scales.
While the straight line method shows an overall trend on the data Which could be inappropriate like in in Fig 3-5.
"""


# What are residuals? Why is it a good idea to plot the residuals of your fit?
"""
According to DAOST residuals (or fitting deviations) are: 
"the remainder when you subtract the smooth “trend” from the actual data"
[ref: DAOST CH3]

It's a good idea to plot residuals to see if they are 
    -symmetrically distributed around zero
    -free of a trend
"""

# Explain in your own words the point of the smooth tube in figure 3-7.
"""
The smooth tube can be interpreted as “confidence bands” which means that most of the points from the data set will 
lie between the two dashed lines. Any point that is out of the tube might be an outlier.    
"""

# What kind of relationships will a semi-log plot help you discover?
"""
A semi-log graph is useful when graphing exponential functions 
[ref : https://nool.ontariotechu.ca/mathematics/basic/points-and-graphs/semi-log-and-log-log-graphs.php#:~:text=In%20a%20semi%2Dlog%20graph,useful%20when%20graphing%20exponential%20functions.]
"""

# What kind of functions will loglog plots help you see?
"""
When one variable changes as a constant power of another, a log-log graph shows the relationship as a straight line. 
[ref: https://statisticsbyjim.com/regression/log-log-plots/]
"""

# What the h#ll is banking and 
# what part of our visual system does it use to help us see patterns? 
# What are potential problems with banking?
"""
- Banking to 45 degrees means to adjust the aspect ratio of the entire plot in such a way that most slopes are at an approximate 45
degree angle. 
[ref: DAOST CH3]

- The purpose of banking is to improve human perception of the graph.
[ref: DAOST CH3]

- When shrinking one axis down this might cause some loses in the details. 
Moreover, if the aspect ratio required to achieve proper banking is too skewed 
it would violate the great affinity humans seem to have for proportions of roughly 4 by 3.
[ref: DAOST CH3]
"""

# I think figure 3-14 makes an important point about linear fits that is rarely made. What is it?
"""
obtaining a different solution when regressing x on y
than when regressing y on x
[ref: DAOST CH3]
"""

# Summarize the discussion of Graphical Analysis and Presentation Graphics on pp. 68-69 in your own words.
"""
Graphical analysis is to investigate data using graphical methods to discover new knowledge and ask proper questions as our understanding of the data evolve.

Presentation graphics is how to communicate the discovered information and results.

When start working on a data-set, at the graphical analysis stage, it's a good practice to draw lots of graphs to better understand the data.
At this stagem it's also recommended to do graph decoration (labels, arrows, special symbols)
as this might be time consuming and we end up throwing the graph as our knowledge about the data and the problem evolve. 
However, the the opposite applies, at presentation graphics stage. 

Some to consider when preparing Presentation graphics:
    - Don’t rely on a caption for basic information. Place basic information on the graph itself.
    - Show Axis labels with unites.
    - Make labels self-explanatory.
    - Pick a suitable font for text on a graph.
    - If exists, explain error bars meaning and choose an appropriate measure of uncertainty.
    - Make sure that data is not unnecessarily obscured by labels
    - Choose appropriate plot ranges
    - Proofread graphs
    - choose an appropriate output format (PDF)
    - Taste can be acquired and learn from others' graphs.

[ref: DAOST CH3]
"""