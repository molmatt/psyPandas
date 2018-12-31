# psyPandas
A high level pandas add-on to make dealing with psychological data quick and simple.

Current version 1.0

Comprised of two classes psyMeasure and psyDataset. psyMeasure is a class that casts portions of pandas DataFrames into an object that then handles all of the routine data conditioning, including: missing data, outliers, reverse keying, and scale scoring. psyDataset isan object that casts the entire dataset into multiple psyMeasures. psyDataset can treat all of the measures similarly and quickly, or each measure can be specified seperately. psyDataset uses all of the specifications that are present in psyMeasures, but also incorporates careless responding screening. 

## psyMeasure
psyMeasure is a class used for making and manipulating psychological measures from pandas dataframes.
                
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
                       
## psyDataset
psyDataset is a class composed of many psyMeasures, for the purpose of scoring and cleaning psychological datasets
    
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
