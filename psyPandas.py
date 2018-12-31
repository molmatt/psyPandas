import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
%matplotlib inline

class psyMeasure():
    """ psyMeasure is a class used for making and manipulating psychological measures from pandas dataframes.
                
        Arguments
        
        data: Takes a pandas dataframe containing the columns that will be used
        
        columns: a list of the columns in the pandas dataframe that should be used when creating the measure, 
                 default is 'all' which uses all of the columns in the dataframe.
                 
        reverseKey: Takes a list of the columns local to psy measure e.g., column 0 = the first column in the measure
                     reverse keys the columns provided according to the anchors provided in the anchors argument.
                     
        scoring: Takes a string and is used to determine the aggregation method to be used for aggregating item level 
                 data to the scale level. Available arguments currently include 'mean', 'sum', and 'factor'.
                     'mean': scores according to the arithmetic mean of the items
                     'sum': scores according to the sum of the items
                     'z' : scores according to the z score of the summed items (i.e., (x-mean(x))/sd(x))
                     'factor': scores according to the factor score (i.e., the factor loading weighted average) corrected
                               for directionality (i.e., will correlate positively with other scoring methods)
                               
        anchors: Takes either a tuple containing two integer values, or the default term, 'infer'. Used for determining 
                 fuctions based on the anchors as well as describing the measure in a more complete manner. 'infer'
                 infers the anchors by taking the minimum and maximum item responses for the entire scale and assumes
                 that those are the anchors. If concerned, pass the tuple argument instead.
                 
        missImp: Takes a string which determines what should be done with missing data in rows that are not missing
                 beyond the tolerance proportion. Default is row mean imputation.
                     'mean': imputes the row mean
                     'median': imputes the row median
                     'mode': imputes the row mode, in the event of two modes the median of them will be chosen
                     'imean': imputes the item mean
                     'imedian': imputes the item median
                     'imode': imputes the item mode, in the event of two modes the median of them will be chosen
                     
        missTol: Takes a proportion. If a larger proportion of this row is missing than the tolerance level, the row
                 will be deleted. if the missingness proportion is lower than the tolerance, the missing data will be
                 imputed according to the method specified in missImp. Default is .5.
                 
        scaleName: Takes a string to serve as the official name of the scale. If left blank the default 'Scale' will 
                   be used.
                   
        outsVal: Takes a float or integer. Entry serves as the critical value for determining the cutoff point for
                 outlying data. default is 3.29, corresponding to a .001 two sided significance.
        
        outsMeth: Takes a string containing either 'z' or 'hi'. Denotes the method to be used for determining outliers. 
                  Currently takes two possible arguments. 'z' finds outliers by the z score method. 'hi' finds outliers
                  using hampel identifiers (i.e., (x - median(x))/median absolute deviation * .675). The .675 is just a
                  constant included to make the scaling more similar to Z.
                
        
        
        
        Attributes
        
        data: returns a pandas DataFrame containing the item level data.
        
        rev_cols: returns a list containing the index of the columns that are being reverse keyed.
        
        score: returns a pandas series of the aggregated scale score.
        
        scale_variance: returns the variance of the scale.
        
        item_variances: returns a pandas series containing the variances of each item
        
        alpha: returns the alpha of the scale for examining the internal consistency of the scale.
        
        full_data: returns a pandas dataframe with the item level data followed by the aggregated scale score.
        
        Methods
        
        itr: returns a pandas series of the item total correlations for each item
        
        citr: returns a pandas series of the item total correlations after correcting for the relationship inflation
              caused by including the item of interest in the total.
              
        factorLoadings: return a pandas DataFrame containing the raw factor loadings in the first column and the 
                        standardized factor loadings in the second column
                        
        alphaIfItemDel: returns a pandas series containing the alpha if the item were deleted
        
        varIfItemDel: returns a pandas series containing the scale variance if the item were deleted
        
        psychometrics: returns a pandas dataframe containing all relevant psychometrics, including:
                           mean : item mean
                           sd : item standard deviation
                           var : item variance
                           min : item minimum response
                           max : item maximum response
                           ITr : item total correlation
                           CITr : corrected item total correlation
                           rawLoadings : raw factor loadings
                           stdLoadings : standardized factor loadings
                           varIfItemDel : variance if item deleted
                           RelIfItemDel : reliability if item deleted
                           
        dropByRel: takes a reliability threshold argument, default is .7. Drops items from scale according to largest 
                   increase in alpha until reliability meets or exceeds threshold.
                   
        dropByCITr: takes a CITr threshold agrument, default is .5. Drops items from scale according to lowest CITr
                    until scale is composed solely of items above the threshold.
                    
        dropByLoading: takes a factor loading threshold agrument, default is .5. Drops items from scale according to 
                       lowest factor loading until scale is composed solely of items above the threshold.
        """
    def __init__(self, data, columns = 'all', scoring = 'mean', reverseKey = 'infer', anchors = 'infer', 
                 missImp = 'mean', missTol = .5, scaleName = 'Scale', outsMeth = 'z', outsVal = 3.29, 
                 verbose = True):
        
        self._data = data
        self._columns = columns
        self._reverseKey = reverseKey
        self._scoring = scoring
        self._anchors = anchors
        self._missImp = missImp
        self._missTol = missTol
        self._scaleName = scaleName
        self._outsMeth = outsMeth
        self._outsVal = outsVal
        
        if self._anchors == 'infer':
            self._anchors = (self.data.min().min(), self.data.max().max())
        if columns == 'all':
            self._data = data.copy()
        elif type(columns) == list:
            self._data = self._data.iloc[:,columns].copy()
        else:
            raise Exception('Invalid column selection')
        if self._reverseKey == 'infer':
            self._reverseKey = []
            for i in self._data:
                if self._data.drop(i, axis = 1).mean(axis=1).corr(self._data[i]) < 0:
                    self._reverseKey.append(self._data.columns.get_loc(i))
        elif not isinstance(self._reverseKey, list):
            raise Exception('Invalid reverse keying argument')    
        self._data.iloc[:,self._reverseKey] = self._data.iloc[:,self._reverseKey].apply(
        lambda x: self._anchors[1]-self._anchors[0]+2-x)

        if self._missImp == 'mean':
            self._data.loc[self._missTol > self._data.isnull().mean(axis = 1),:] = self._data.T.fillna(self._data.mean(axis = 1)).T
            self._data.dropna(axis = 0, inplace = True)
        elif self._missImp == 'median':
            self._data.loc[self._missTol > self._data.isnull().mean(axis = 1),:] = self._data.T.fillna(self._data.median(axis = 1)).T
            self._data.dropna(axis = 0, inplace = True)
        elif self._missImp == 'mode':
            self._data.loc[self._missTol > self._data.isnull().mean(axis = 1),:] = self._data.T.fillna(self._data.mode(axis = 1).median()).T
            self._data.dropna(axis = 0, inplace = True)
        elif self._missImp == 'imean':
            self._data.loc[self._missTol > self._data.isnull().mean(axis = 1),:] = self._data.fillna(self._data.mode().median())
            self._data.dropna(axis = 0, inplace = True)
        elif self._missImp == 'imedian':
            self._data.loc[self._missTol > self._data.isnull().mean(axis = 1),:] = self._data.fillna(self._data.median())
            self._data.dropna(axis = 0, inplace = True)
        elif self._missImp == 'imode':
            self._data.loc[self._missTol > self._data.isnull().mean(axis = 1),:] = self._data.fillna(self._data.mode().median())
            self._data.dropna(axis = 0, inplace = True)
        elif self._missImp == 'regression':
            regdic = {}
            for i in self._data.columns:
                regdic[i] = []
            for i, r in self._data.iterrows():
                if (any(r.isna()) and r.isna().mean() < self._missTol):
                    mi = self._data.loc[:,r.isna()]
                    mi = mi.loc[mi.isna().mean(axis = 1) == 1,:]
                    y = self._data.loc[:,r.isna()].copy()
                    y.dropna(axis = 0, inplace = True)
                    x = self._data.loc[:,~r.isna()].copy()
                    x.dropna(axis = 0, inplace=True)
                    tra = x.merge(y, how='inner', left_index=True, right_index=True)
                    xs = len(x.columns)
                    lr = LinearRegression().fit(X=tra.iloc[:,:xs],y=tra.iloc[:,xs:])
                    pre = x.merge(mi, left_index=True, right_index=True)
                    im = lr.predict(pre.iloc[:,:xs])
                    for k in range(len(im[0])):
                        regdic[mi.columns[k]].append(im[0][k])
                    for k in x.columns:
                        regdic[k].append(np.NaN)
                else:
                    for j in self._data.columns:
                        regdic[j].append(np.NaN)
            self._data.fillna(pd.DataFrame(regdic),inplace=True)
            self._data.dropna(axis = 0, inplace=True)
        if self._outsMeth != 'keep':
            a=(self._data.sum(axis=1))
            if self._outsMeth == 'z':
                self.outliers = np.abs((a - a.mean())/a.std()) < self._outsVal
            elif self._outsMeth == 'hi':
                self.outliers = np.abs((a - a.median())/(.675*np.median(np.abs(a - a.median())))) < self._outsVal
            else:
                raise Exception('Invalid outlier detection method chosen')
            self._data = self._data.loc[self.outliers,:]
            
        if verbose == True:
            if reverseKey == 'infer':
                print('Inferred Reverse Keys:', self._data.columns[self._reverseKey].tolist())
            if anchors == 'infer':
                print('Inferred Anchors:',self._anchors)
        
        self.rev_cols = self._data.columns[self._reverseKey]
            
        self._items = len(self.data.columns)
        
        self._itemMeans = self.data.mean()

        self._scaleVariance = np.var(self.score)
        
        self._itemVariances = self.data.var()
        
        #self._itemStdDev = self.data.std()
        
        self._alpha = (self.items/(self.items-1))*(1-(sum(self.itemVariances)/np.var(self.data.sum(axis=1))))
        
        self._fullData = self.data.copy()
        self._fullData['Scale'] = self.score
        
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        
    @property
    def stdData(self):
        return (self._data-self._data.mean())/self._data.std()
        
    @property
    def score(self):
        if self._scoring == 'mean':
            self._score = self._data.mean(axis=1)
        elif self._scoring == 'sum':
            self._score = self._data.sum(axis=1)
        elif self._scoring == 'z': 
            self._score = (self._data.sum(axis=1)-self._data.sum(axis=1).mean())/self._data.sum(axis=1).std()
        elif self._scoring == 'factor':
            self._score = pd.Series([s[0] for s in FactorAnalysis(n_components=1).fit_transform(self._data)])
            if np.corrcoef(self._score, self._data.mean(axis=1))[0,1] < 0:
                self._score *= -1
        else:
            raise Exception('Invalid scoring method')
        return self._score
            
    @score.setter
    def score(self, value):
        self._score = value
        
    @property
    def items(self):
        return len(self.data.columns)
    
    @items.setter
    def items(self, value):
        self._items = value
        
    @property
    def itemMeans(self):
        return self.data.mean()
        
    @property
    def scaleVariance(self):
        return self.score.var() 
        
    @property
    def itemVariances(self):
        return self.data.var()
        
    @property
    def itemStdDev(self):
        return self.data.std()
        
    @property
    def itemMins(self):
        return self.data.min()
    
    @property
    def itemMaxs(self):
        return self.data.max()
        
    @property
    def alpha(self):
        return (self.items/(self.items-1))*(1-(sum(self.itemVariances)/np.var(self.data.sum(axis=1))))
        
    @property
    def fullData(self):
        self._fullData = self.data.copy()
        self._fullData[self._scaleName] = self.score
        return self._fullData
    
    @fullData.setter
    def fullData(self, value):
        if isinstance(value, pd.DataFrame):
            self._fullData = value
        else:
            raise Exception("fullData must be in the form of a pandas DataFrame, currently :", value.__class__)
        
    def ITr(self):
        '''
        Returns a pandas series of the correlations of each item with the scale score.
        
        The ITr will likely be an inflated estimate of the items relationships with the scale. CITr's correct for this
        by removing the item from the calculation of the scale score before correlating each item with it. CITrs are
        implemented as a method which can be called using .CITr().
        '''
        return self.fullData.corr().iloc[:-1,-1]
        
    def CITr(self):
        '''
        Returns a pandas series of the correlations of each item with the scale score corrected for the items role
        in creating the scale score.
        
        Each CItr has the item removed from the calculation of the scale score. This should in most cases be used
        rather than the .ITr() method. This also stores the results in an attribute .citr.
        '''
        self.citr = []
        for i in self.data:
            self.citr.append(np.corrcoef(self.data.drop(i, axis = 1).mean(axis=1), self.data[i])[0,1])
        self.citr = pd.Series(data = self.citr, index = self.data.columns)
        return self.citr
    
    def factorLoadings(self):
        '''
        Returns a pandas dataframe containing the raw and standardized factor loadings of each item on a single factor.
        
        This method provides the unstandardized "rawLoadings", and the standardized "stdLoadings" for the items on a
        single factor, using scikit-learn's FactorAnalysis algorithm. This is used for determining which items fit best
        with the construct. 
        '''
        return  pd.DataFrame({
            'rawLoadings' : pd.Series(FactorAnalysis(n_components=1).fit(self._data).components_[0], 
                                      index=self.data.columns),
            'stdLoadings' : pd.Series(FactorAnalysis(n_components=1).fit(self.stdData).components_[0], 
                                      index=self.data.columns)
            })

    def alphaIfItemDel(self):
        '''
        Returns a pandas series containing the alpha if item deleted.
        
        This method provides the alphas if items deleted for each item in the scale. Note these are unstandardized alphas.
        This can be useful when making decisions about which items to drop in order to achieve acceptable reliatbility.
        This method is used by the psychometrics(), dropByRel(), and dropManually() methods. 
        '''
        alphaIfItemDel = []
        for i in self.data:
            a = len(self.data.drop(i, axis=1).columns)
            alphaIfItemDel.append(
                (a/(a-1))*(1-(sum(self.data.drop(i, axis = 1).var())/(np.var(self.data.drop(i, axis = 1).sum(axis=1))))))        
        return pd.Series(alphaIfItemDel, index = self.data.columns)
    
    def varIfItemDel(self):
        '''
        Returns a pandas series containing the scale's variance if each item were deleted.
        
        This method provides what the scale's variance would be if each item were deleted. This is also displayed 
        when the psychometrics() method is called.
        '''
        varIfItemDel = []
        for i in self.data:
            if self._scoring == 'mean':
                a = self._data.drop(i, axis=1).mean(axis=1)
            elif self._scoring == 'sum':
                a = self._data.drop(i, axis=1).sum(axis=1)
            elif self._scoring == 'z': 
                a = (self._data.drop(i, axis=1).sum(axis=1)-self._data.drop(i, axis=1).sum(axis=1).mean())/(
                    self._data.drop(i, axis=1).sum(axis=1).std())
            elif self._scoring == 'factor':
                a = pd.Series([s[0] for s in FactorAnalysis(n_components=1).fit_transform(self._data.drop(i, axis=1))])
            varIfItemDel.append(a.var())
        return pd.Series(varIfItemDel, index = self.data.columns)
    
    def psychometrics(self):
        '''
        Returns a variety of psychometric information that is useful for describing the items that compose scales.
        '''
        return pd.DataFrame({
            'mean' : self.itemMeans.append(pd.Series(self.score.mean(),index = [self._scaleName])),
            'sd' : self.itemStdDev.append(pd.Series(self.score.std(),index = [self._scaleName])),
            'var' : self.itemVariances.append(pd.Series(self.score.var(),index = [self._scaleName])),
            'min' : self.itemMins.append(pd.Series(self.score.min(),index = [self._scaleName])),
            'max' : self.itemMaxs.append(pd.Series(self.score.max(),index = [self._scaleName])),
            'ITr' : self.ITr(),
            'CITr' : self.CITr(),
            'rawLoadings' : self.factorLoadings().rawLoadings,
            'stdLoadings' : self.factorLoadings().stdLoadings,
            'varIfItemDel' : self.varIfItemDel(),
            'alphaIfItemDel' : self.alphaIfItemDel()
            })
    
    def dropByRel(self, thresh = .7, verbose = False):
        '''
        Drops items from the scale iteratively according to the greatest increase in alpha. Continues dropping items 
        until the alpha is as or above the threshold specified in the 'thresh' argument. Default threshold is .7.
        
        Arguments
        thresh: the reliability threshold for determining when item dropping will cease.
        verbose: if True, will display the psychometrics after each iteration of item dropping.
        '''
        while self.alpha <= thresh:
            print(self.data.columns[self.alphaIfItemDel().values.argmax()],
                  ":",
                  self.alphaIfItemDel().values.max())
            self._data.drop(self._data.columns[self.alphaIfItemDel().values.argmax()], 
                            axis = 1, inplace = True)
            if verbose:
                display(self.psychometrics())
            if self.items == 2:
                raise Exception('Threshold cannot be reached')
            
    def dropByCITr(self, thresh = .5, absolute = False, verbose = False):
        '''
        Drops items from the scale iteratively according to whether their CITr is below the threshold. Continues dropping
        items until all item CITrs are at or above the threshold.
        
        Arguments
        thresh: the CITr threshold. All items under this threshold will be dropped.
        absolute: determines whether the absolute value of the CITrs should be used. If True, absolute values are used,
                  if False, raw values are used.
        verbose: if True, will display the psychometrics after each iteration of item dropping.
        '''
        if absolute:
            a = np.abs(self.CITr())
        else:
            a = self.CITr()
        while a.min() <= thresh:
            if absolute:
                a = np.abs(self.CITr())
            else:
                a = self.CITr()
            print(self.data.columns[a.values.argmin()],
                  ":",
                  a.values.min())
            self._data.drop(self._data.columns[a.values.argmin()], 
                            axis = 1, inplace = True)
            if self.items == 2:
                raise Exception('Threshold cannot be reached')
            if verbose:
                display(self.psychometrics())
            if absolute:
                a = np.abs(self.CITr())
            else:
                a = self.CITr()
            
    def dropByLoading(self, thresh = .5, absolute = True, verbose = False):
        '''
        Drops items from scale according to whether their standardized factor loadings are below the specified threshold.
        
        thresh: factor loading threshold under which items will be dropped
        absolute: determines whether the absolute standardized factor loading should be used, or the factor loadings,
                  including negatives. If True uses absolute.
        verbose: if True, displays psychometrics at each iteration of item dropping.
        '''
        if absolute:
            a = np.abs(self.factorLoadings().stdLoadings)
        else:
            a = self.factorLoadings().stdLoadings
        while a.min() <= thresh:
            if absolute:
                    a = np.abs(self.factorLoadings().stdLoadings)
            else:
                    a = self.factorLoadings().stdLoadings
            print(self._data.columns[a.values.argmin()],
                  ":",
                  a.values.min())
            self._data.drop(self._data.columns[a.values.argmin()], 
                            axis = 1, inplace = True)
            if self.items == 2:
                raise Exception('Threshold cannot be reached')
            if verbose:
                display(self.psychometrics())
            if absolute:
                a = np.abs(self.factorLoadings().stdLoadings)
            else:
                a = self.factorLoadings().stdLoadings

    def dropManually(self):
        '''
        Displays psychometrics and provides a user input prompt. Providing the name of an item will result in that item
        being dropped and the psychometrics being updated. This continues until the user types, 'done'.
        '''
        print('Type the name of the item to drop, when finished type "done"')
        display(self.psychometrics())
        action = input()
        while action != 'done':
            self._data.drop(action, axis = 1, inplace = True)
            display(self.psychometrics())
            action = input()
            
class psyDataset():
    """ psyDataset is a class composed of many psyMeasures, for the purpose of scoring and cleaning psychological datasets
    
    Arguments
    
    data: Takes a pandas dataframe containing all of the item level data for the dataset from which scales will be made
    
    columns: Takes a list of lists, each sub-list contains the columns for each of the scales. For example if you wanted 
             to make three scales out with the first three items of the dataset in the first scale, the second two items 
             in the second scale and the last 4 items in the third scale the argument would look like this.
             
                 cols = [[0,1,2],
                         [3,4],
                         [5,6,7,8]]
                 psyDataset(data = df, columns = cols, names = nams)
                 
    names: Takes a list of strings that contain the desired names of each scale in the order that the columns were
           designated. For example, lets say the first scale should be called 'Extraversion', the
           second scale should be called 'Agreeableness', and the third scale, 'Conscientiousness'.
           
               nams = ['Extraversion',
                       'Agreeableness',
                       'Conscientiousness']
               psyDataset(data = df, columns = cols, names = nams)
               
    scoring: Takes either a single string if scoring is consistent across all scales, or a list of strings if scoring
             differs for each scale. This is what will be used to determine how to aggregate data from the item level to
             the scale level. Supported aggregation methods currently include:
                 'mean': scores according to the arithmetic mean of the items, this is the default option
                 'sum': scores according to the sum of the items
                 'z' : scores according to the z score of the summed items (i.e., (x-mean(x))/sd(x))
                 'factor': scores according to the factor score (i.e., the factor loading weighted average) corrected
                           for directionality (i.e., will correlate positively with other scoring methods)
             For example, lets say we wanted the first scale to be scored using means, the second to be scored using
             z scores, and the third to be a factor score.
                 
                 sco = ['mean',
                        'z',
                        'factor']
                 psyDataset(data = df, columns = cols, names = nams, scoring = sco)
               
    reverseKeys: Takes a list of lists that contain the columns (local to each scale) that should be reverse keyed. Empty
                 lists should be included for scales that don't have any reverse keys. If there are no reverse keyed
                 items in the dataset leave this argument blank. Remember the reverse keying columns are local to that
                 scale. So if this scale starts on column 5, and goes to column 10, and you want to reverse key the first
                 and fourth item in the scale, the reverse key list should look like [0,3]. Carrying on our example from
                 above, lets say we wanted no reverse keys in the first scale, the second item of the second scale reverse
                 keyed, and the first and third item of the the third scale to be reverse keyed. 
                 
                     rk = [[],
                           [0],
                           [0,2]]
                     psyDataset(data = df, columns = cols, names = nams, reverseKeys = rk)
                     
    anchors: Takes either the string 'infer', one tuple of anchors for the entire dataset (e.g., 1,5), or a list of
             of tuples containing the anchors for each scale. 'infer' takes the minimum item response for the scale, and
             sets that to the lower anchor, and takes the maximum item response for the scale and sets that to the upper
             anchor. For example, if the first scale is on a 7 point, the second on a 5, and the third on a 7. 
                    
                    anc = [(1, 7),
                           (1, 5),
                           (1, 7)]
                    psyDataset(data = df, columns = cols, names = nams, reverseKeys = rk, anchors = anc)
                    
    missImp: Takes a string which determines what should be done with missing data in rows that are not missing
             beyond the tolerance proportion. Default is row mean imputation.
                 'mean': imputes the row mean
                 'median': imputes the row median
                 'mode': imputes the row mode
                 'imean': imputes the item mean
                 'imedian': imputes the item median
                 'imode': imputes the item mode, in the event of multiple modes it takes the median
                 'regression': uses the available data to impute missing data using a regression trained on the present
                               data
             Alternatively, takes a list of strings denoting what should be done with each scale to be measured in the
             psyDataset.
    
    missTol: Takes a proportion. If a larger proportion of this row is missing than the tolerance level, the row
             will be deleted. if the missingness proportion is lower than the tolerance, the missing data will be
             imputed according to the method specified in missImp. Default is .5. Alternatively, takes a list of
             proportions to be used for each scale.
    
    outsMeths: Takes a string containing either 'z' or 'hi'. Denotes the method to be used for determining outliers. 
               Currently takes two possible arguments. 'z' finds outliers by the z score method. 'hi' finds outliers
               using hampel identifiers (i.e., (x - median(x))/median absolute deviation * .675). The .675 is just a
               constant included to make the scaling more similar to Z.
    
    outsVals: Takes a float or integer. Entry serves as the critical value for determining the cutoff point for
              outlying data. default is 3.29, corresponding to a .001 two sided significance.
    
    CRItems: Takes a list containing the columns that are being used to identify careless responders.
    
    CRKey: Takes a list containing the correct answers for the careless responding items. 
    
    CRTol: The number of careless responding items that can be missed before the case is considered a careless
           responder. Default value is 0.
    
    """
    def __init__(self, data, columns, names, reverseKeys = 'infer', anchors = 'infer', scoring = 'mean',  
                 missImp = 'mean', missTol = .5, outsMeths = 'z', outsVals = 3.29, CRItems = [], 
                 CRKey = [], CRTol = 0):
        self._data = data
        self._columns = columns
        self._names = names
        self._reverseKeys = reverseKeys
        self._scoring = scoring
        self._anchors = anchors
        self._missImp = missImp
        self._missTol = missTol
        self._outsMeths = outsMeths
        self._outsVals = outsVals
        self._CRItems = CRItems
        self._CRKey = CRKey
        self._CRTol = CRTol
        self._data = self._data.loc[self.CRScore <= self._CRTol,:]

        for i in range(len(self._names)):
            setattr(self, self._names[i], 
                    psyMeasure(data = self._data, 
                               columns = self._columns[i], 
                               scaleName = self._names[i],
                               reverseKey = self._reverseKeys if isinstance(self._reverseKeys,str) else self._reverseKeys[i],
                               scoring = self._scoring if not isinstance(self._scoring, list) else self._scoring[i],
                               anchors = self._anchors if isinstance(self._anchors, str) else self._anchors[i],
                               missImp = self._missImp if isinstance(self._missImp, str) else self._missImp[i],
                               missTol = self._missTol if not isinstance(self._missTol, list) else self._missTol[i],
                               outsMeth = self._outsMeths if isinstance(self._outsMeths, str) else self._outsMeths[i],
                               outsVal = self._outsVals if not isinstance(self._outsVals, list) else self._outsVals[i]
                               ))
        
    @property
    def scores(self):
        self._scores = {}
        for n in self._names:
            self._scores[n] = getattr(self, n).score
        self._scores = pd.DataFrame(self._scores)
        return self._scores
    
    @property
    def CRScore(self):
        self._CRScore = pd.Series(np.zeros(len(self._data)))
        for i in range(len(self._data)):
            for c, k in zip(self._CRItems, self._CRKey):
                if self._data.iloc[i,c] != k:
                    self._CRScore[i] += 1
        return self._CRScore
    
    @property
    def alphas(self):
        self._alphas = []
        for n in self._names:
            self._alphas.append(getattr(self, n).alpha)
        self._alphas = pd.Series(self._alphas, index=self._names)
        return self._alphas
    
    @property
    def summary(self):
        a = pd.DataFrame(self.scores.mean(),columns=['Mean'])
        b = pd.DataFrame(self.scores.std(),columns=['SD'])
        c = self.scores.corr()
        for i in range(len(c)):
            c.iloc[i,i] = self.alphas[i]
        self._summary = a.merge(b, left_index=True, right_index=True).merge(c,left_index=True, right_index=True)
        return self._summary
                    