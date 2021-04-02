# confMatrixKF = confusion_matrix(targets_true, targets_pred)
# confMatrixKFpct = confMatrixKF/np.sum(confMatrixKF,axis=1).reshape(-1,1)
# #when applying .ravel() to confusion matrix the order is: TN, FP, FN, TP

# cm = ['{0:0.0f}'.format(val) for val in confMatrixKF.flatten()]
# cmpct = ['{0:.2%}'.format(val) for val in confMatrixKFpct.flatten()]
# labels = [f'{val1}({val2})' for val1, val2 in zip(cm,cmpct)]
# labels = np.asarray(labels).reshape(nCategories,nCategories)
    
# title = ('Crime Prediction using Random Forest Classifier'
#          +'\nPerformance on 10-Fold Cross Validation'
#          +f'\nCrime Categories: {crime1}(0), {crime2}(1)'
#          +f'\nDataset: {minCategoryCount} randomly selected instances per category'
#          +f'\nClassifier Parameters: nTrees:{nTrees}, maxDepth:{max_depth}, min_leaf_size:{min_leaf_size}'
#          +f'\ncriterion:{criterion}, randomState:{random_state}')

# plt.figure()
# sn.heatmap(confMatrixKFpct, annot=labels, fmt='', cmap="Blues")
# plt.xlabel('Predicted Crime Categories')
# plt.ylabel('True Crime Categories')
# plt.title(title)
# plt.show()
