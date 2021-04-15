"""Run basic metrics on DDFs"""
import pandas as pd
import numpy as np
import os, sys

from notify_run import Notify
notify = Notify()

# import lsst.sim.maf moduels modules
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
from lsst.sims.maf.stackers import BaseStacker
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles

# import actual metric class
# from AGNMetrics import DCRPrecisionMetric
# from AGNStacker import MagErrStacker

# import convenience functions
from opsimUtils import *

# import joblib
from joblib import Parallel, delayed

ddfFields = ['COSMOS', 'XMM-LSS', 'ELAISS1', 'ECDFS', 'EDFS']

# define function to run MAF on one opsim which is easily parallelziable. 
def run_sf_ddf(run, src_mags, dbDir, outDir, metricDataPath, **kwargs):
    """
    Function to run SFErrorMetric on one OpSim. 
    
    Args:
        run (str): The OpSim cadence run name.
        src_mags (list): At what source magnitudes to evaluate, keys are bands. 
            Defaults to {'u': [24.15], 'g': [24]}
        dbDir (str): The path to the OpSim databases.
        outDir (str): The path to the resultdb databases.
        metricDataPath (str): The path to the actual metric data (.npz files). 
    """
    
    rt = ''
    try:
        # init connection given run name
        opSimDb, resultDb = connect_dbs(dbDir, outDir, dbRuns=[run])

        # if no DDF return
        prop_info = opSimDb[run].fetchPropInfo()
        if (not 'DD' in prop_info[1]) or (len(prop_info[1]['DD']) == 0):
            print(f'No DDF for {run}')
            return rt

        # init bundleDict
        bundleDict = {}

        # shared configs
        slicer = slicers.HealpixSlicer(nside=128)
        base_constraint = 'filter = "{}"'
        summaryMetrics = [metrics.MedianMetric()]

        # loop through bands and source mags to init metricBundle
        for band in src_mags:
            mags = src_mags[band]

            # loop through each DDF
            for ddf in ddfFields:                
                proposalIds = ddfInfo(opSimDb[run], ddf)['proposalId']

                # ddf constraint based on number of fields in opsim
                if len(proposalIds) > 1:
                    ddf_constraint = base_constraint.format(band) + \
                                    f" and (proposalId = {proposalIds[0]}" + \
                                    f" or proposalId = {proposalIds[1]})"
                else:
                    ddf_constraint = base_constraint.format(band) + \
                                    f" and proposalId = {proposalIds[0]}"
                    
                ## m5 metric
                ## -------------------------------------------------------------------
                # - p25
                m5p25 = metrics.PercentileMetric('fiveSigmaDepth', 
                                                metricName = f'm5p25_{band}_{ddf}')
                m5p25_mb = metricBundles.MetricBundle(m5p25, slicer, ddf_constraint)
                bundleDict[m5p25.name] = m5p25_mb
                
                # - p50
                m5Median = metrics.MedianMetric('fiveSigmaDepth', 
                                                metricName = f'm5Median_{band}_{ddf}')
                m5Median_mb = metricBundles.MetricBundle(m5Median, slicer, ddf_constraint)
                bundleDict[m5Median.name] = m5Median_mb
                
                # - p75
                m5p75 = metrics.PercentileMetric('fiveSigmaDepth', 
                                                metricName = f'm5p75_{band}_{ddf}')
                m5p75_mb = metricBundles.MetricBundle(m5p75, slicer, ddf_constraint)
                bundleDict[m5p75.name] = m5p25_mb
                ## -------------------------------------------------------------------
                
                ## airmass metric
                airmassMax = metrics.MaxMetric('airmass', 
                                               metricName = f'airmassMax_{band}_{ddf}')
                airmassMax_mb = metricBundles.MetricBundle(airmassMax, slicer, ddf_constraint)
                bundleDict[airmassMax.name] = airmassMax_mb

                ## nvisit
                nvisit = metrics.CountMetric('observationStartMJD', 
                                             metricName = f'nvisit_{band}_{ddf}')
                nvisit_mb = metricBundles.MetricBundle(nvisit, slicer, ddf_constraint)
                bundleDict[nvisit.name] = nvisit_mb
                
                ## coadd
                coadd = metrics.Coaddm5Metric('fiveSigmaDepth', 
                                             metricName = f'coadd_{band}_{ddf}')
                coadd_mb = metricBundles.MetricBundle(coadd, slicer, ddf_constraint)
                bundleDict[coadd.name] = coadd_mb

        # set runname and summary stat
        for key in bundleDict:
            bundleDict[key].setRunName(run)
            bundleDict[key].setSummaryMetrics(summaryMetrics)
    

        # make a group
        metricGroup = metricBundles.MetricBundleGroup(bundleDict, opSimDb[run], 
                                                      metricDataPath, 
                                                      resultDb[run], verbose=False)
        metricGroup.runAll()

        # close dbs
        opSimDb[run].close()
        resultDb[run].close()

    except Exception as e:
        print(f'{run} failed!')
        print(e)
        print('----------------------')
        rt = run
    
    return rt


# function to run entire fbs version
def run_fbs(version, dbDir, outDir, metricDataPath):
    
    # create if not exists
    if not os.path.exists(os.path.abspath(outDir)):
        os.makedirs(os.path.abspath(outDir))
    
    if not os.path.exists(os.path.abspath(metricDataPath)):
        os.makedirs(os.path.abspath(metricDataPath))
            
    # get all runs
    dbRuns = show_opsims(dbDir)[:]
        
    # define metric parameters for DDF (not used here)
    src_mags = {'u': [22.15], 'g': [22], 'r': [21.75], 
                'i': [21.65], 'z': [21.55], 'y': [21.45]}

    # placeholder for joblib returned result
    rt = []
    rt = Parallel(n_jobs=14)(delayed(run_sf_ddf)(run, src_mags, dbDir, 
                                                 outDir, metricDataPath)
                             for run in dbRuns)

    # check failed 
    failed_runs = [x for x in rt if len(x) > 0]
    
    # rerun failed ones caused sql I/O error
    for run in failed_runs:
        print(f'Rerun failed: {run}')
        print('-------------------------------------')
        try:
            run_sf_ddf(run, src_mags, dbDir, outDir, metricDataPath)
            failed_runs.remove(run)
        except:
            continue
    
    if len(failed_runs) > 0:
        with open(f'v{version}_DDF.log', 'a') as f:
            for run in failed_runs:
                f.write(run+'\n')

    notify.send(f"Done with DCR_DDF FBS_v{version}!")
    
    
if __name__ == "__main__":
    
    # FBS versions to run
    versions = ['1.5', '1.6', '1.7']
    
    # get input from command line
    dbDir_temp = '/home/idies/workspace/lsst_cadence/FBS_{}/'
    outputFolder = sys.argv[1]
    
    outDir = os.path.join(outputFolder, 'ResultDBs')
    metricDataPath = os.path.join(outputFolder, 'MetricData')
    
    for version in versions[:]:
        dbDir = dbDir_temp.format(version)
        run_fbs(version, dbDir, outDir, metricDataPath)