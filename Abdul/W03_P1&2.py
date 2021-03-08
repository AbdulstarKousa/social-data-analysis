def open_in_chrome(Video_id):
    import webbrowser

    url_base = 'https://youtu.be/'

    webbrowser.register(
        'chrome',
        None,
        webbrowser.BackgroundBrowser("C://Program Files (x86)//Google//Chrome//Application//chrome.exe")
        )

    webbrowser.get('chrome').open(url_base + Video_id)




############################################
# Part 1: Fundamentals of data visualization
############################################
# Excercise: Questions for the lecture of Part 1.1
open_in_chrome('9D2aI30AMhM')
"""
* What is the difference between data and metadata? How does that relate to the bike-example?
  see https://dataedo.com/kb/data-glossary/what-is-metadata
* Sune says that the human eye is a great tool for data analysis. Do you agree? Explain why/why not. 
  Mention something that the human eye is very good at.
  Can you think of something that is difficult for the human eye. 
  Explain why your example is difficult.
  Ans:   
    Easy To understand plots (Ex D02 Part 01)
* Simpson's paradox is hard to explain. Come up with your own example - or find one on line.
  Ans: 
* In your own words, explain the difference between exploratory and explanatory data analysis.
  Ans:
    See http://www.storytellingwithdata.com/blog/2014/04/exploratory-vs-explanatory-analysis#:~:text=Exploratory%20analysis%20is%20what%20you%20do%20to%20get%20familiar%20with%20the%20data.&text=Exploratory%20analysis%20is%20the%20process,1%20or%202%20precious%20gemstones.
"""


#Excercise: Questions for the lecture of Part 1.2
open_in_chrome('yiU56codNlI')
"""
* As mentioned earlier, visualization is not the only way to test for correlation. 
  We can (for example) calculate the Pearson correlation. 
  Explain in your own words how the Pearson correlation works and write down it's mathematical formulation. 
  Can you think of an example where it fails (and visualization works)?
  Ans:
    See https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
    see https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    The example is from D02 Part 01  


* What is the difference between a bar-chart and a histogram?
  Ans:
    Histograms are used to show distributions of variables 
    while bar charts are used to compare variables. 
    
    Histograms plot quantitative data with ranges of the data grouped into bins or intervals 
    while bar charts plot categorical data

* I mention in the video that it's important to choose the right bin-size in histograms. But how do you do that?
  Do a Google search to find a criterion you like and explain it.
  Ans:
    See: https://www.statisticshowto.com/choose-bin-sizes-statistics/
    - Bins should be all the same size. For example, groups of ten or a hundred.
    - Bins should include all of the data, even outliers. If your outliers fall way outside of your other data,
      consider lumping them in with your first or last bin. This creates a rough histogram —make sure you note where outliers are being included.
    - Boundaries for bins should land at whole numbers whenever possible (this makes the chart easier to read).
    - Choose between 5 and 20 bins. The larger the data set, the more likely you’ll want a large number of bins.
      For example, a set of 12 data pieces might warrant 5 bins but a set of 1000 numbers will probably be more useful with 20 bins. 
      The exact number of bins is usually a judgment call.
    - If at all possible, try to make your data set evenly divisible by the number of bins.
      For example, if you have 10 pieces of data, work with 5 bins instead of 6 or 7.
"""



###################################################
# Part 2: Reading about the theory of visualization
###################################################

# Reading see Week3 DTU learn 

# Excercise: Questions for the reading part
"""
* Explain in your own words the point of the jitter plot.
  Ans:
    Avoid multipile points on the same place that happend on dot plot
* Explain in your own words the point of figure 2-3.
  (I'm going to skip saying "in your own words" going forward,
  but I hope you get the point;
  I expect all answers to be in your own words).
  Ans:
    Choosing Where do we now place the first bin change the chape of the histogram 
    (the alignment of the bins on the x-axis)
  

    
* The author of DAOST (Philipp Janert) likes KDEs
 (and think they're better than histograms).
  And I don't. I didn't give a detailed explanation in the video,
  but now that works to my advantage.
  I'll ask you guys to think about this 
  and thereby create an excellent exercise: 
  When can KDEs be misleading? 
  Ans: 
    The teacher will provide the answer in a later lecture :).
* I've discussed some strengths of the CDF -
  there are also weaknesses.
  Janert writes "CDFs have less intuitive appeal
  than histograms of KDEs". What does he mean by that?
  Ans:

* What is a Quantile plot? What is it good for.
    Ans:
    A quantile plot is just the plot of a CDF in which the x and y-axes have beenswitched

* How is a Probability plot defined? 
  What is it useful for? 
  Have you ever seen one before?
  Ans:
  A P-P plot compares the empirical cumulative distribution function
  of a data set with a specified theoretical cumulative distribution function F(·)

  To see if the data follow a specific distribution 

* One of the reasons I like DAOST is that Janert 
  is so suspicious of mean, median, and related summary statistics. 
  Explain why one has to be careful when using those -
  and why visualization of the full data is always better.
  D 02 

* I love box plots. When are box plots most useful?
  provide a visual summary statistic of the data // and outliers 

* The book doesn't mention violin plots,
  see https://en.wikipedia.org/wiki/Violin_plot,
  Are those better or worse than box plots? Why?

  A violin plot is more informative than a plain box plot.
  While a box plot only shows summary statistics such as mean/median and interquartile ranges,
  the violin plot shows the full distribution of the data.
  The difference is particularly useful when the data distribution is multimodal (more than one peak).
"""

