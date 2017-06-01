import numpy
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable

def ransac(data,model,n,max_iterations,threshold,max_inliers,debug=False,return_all=False):
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None

    # Loop for the specified iterations
    # You could also do it using time
    while iterations < max_iterations:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        #test_points = data[test_idxs]

        # Pick n random points
        #maybe_idxs = numpy.random.choice(data, n)

        # Create the model
        maybemodel = model.fit(maybe_idxs)

        # Check the model for validity
        if not model.is_valid_model(maybemodel, maybe_idxs):
            iterations+=1
            print("Invalid model")
            continue

        # Get the error for the model
        test_err = model.get_error(test_idxs, maybemodel)

        # Invalid error
        if test_err is None:
            #print("Invalid error")
            iterations+=1
            continue

        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers)))

        if len(alsoinliers) > d:
            betterdata = numpy.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = numpy.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2