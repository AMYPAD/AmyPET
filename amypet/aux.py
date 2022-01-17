'''
Auxiliary functions for the centiloid project
'''

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from niftypet import nimpa

# > regions used in CL project
rvois = ['wc', 'cg', 'wcb', 'pns']

# > region full name strings for plots
rvoi_str = dict(
    wc='whole cerebellum',
    cg='cerebellum GM',
    wcb='whole cerebellum + brain stem',
    pns='pons')

#----------------------------------------------------------------------
def im_check_pairs(fpets, fmris):
    '''
    checks visually image by image if the PET and MR have
    the same orientation.
    '''
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    for fp, fm in zip(fpets, fmris):
        print(f'{fp} : {fm}')
        p = nimpa.getnii(fp)
        m = nimpa.getnii(fm)

        ps = p.shape
        ms = m.shape

        ax[0,0].imshow(p[ps[0]//2, ...])
        ax[1,0].imshow(m[ms[0]//2, ...])

        ax[0,1].imshow(p[..., ps[2]//2])
        ax[1,1].imshow(m[..., ms[2]//2])

        plt.draw()
        plt.waitforbuttonpress(0)
#----------------------------------------------------------------------


#----------------------------------------------------------------------
def identity_line(ax=None, ls='--', *args, **kwargs):
    '''
    plot identity line for any correlation analysis
    needed for the CL calibration and beyond.
    '''
    # see: https://stackoverflow.com/q/22104256/3986320
    ax = ax or plt.gca()
    identity, = ax.plot([], [], ls=ls, *args, **kwargs)
    def callback(axes):
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)
    return ax
#----------------------------------------------------------------------


#----------------------------------------------------------------------
def get_clref(fxls):
    '''
    get the excel sheets with CL reference data from the CL project
    excel sheet.
    '''
    
    # > Define variable to load the workbook
    wb = openpyxl.load_workbook(fxls)
    ws = wb.active

    pib_tbl = dict(yc={},ad={})

    # > cell offsets
    ioff_ad = [5,50]
    ioff_yc = [51,85]

    sbj = [str(i.value) for i in ws['A']]
    cg  = [str(i.value) for i in ws['B']]
    wc  = [str(i.value) for i in ws['C']]
    wcb = [str(i.value) for i in ws['D']]
    pns = [str(i.value) for i in ws['E']]

    ccg  = [str(i.value) for i in ws['F']]
    cwc  = [str(i.value) for i in ws['G']]
    cwcb = [str(i.value) for i in ws['H']]
    cpns = [str(i.value) for i in ws['I']]


    # > young control indices and SUVr/CL values
    pib_tbl['yc']['id'] = np.array([int(i[3:]) for i in sbj[ioff_yc[0]:ioff_yc[1]]])

    pib_tbl['yc']['suvr'] = {}
    pib_tbl['yc']['suvr']['cg']  = np.array([float(i) for i in  cg[ioff_yc[0]:ioff_yc[1]]])
    pib_tbl['yc']['suvr']['wc']  = np.array([float(i) for i in  wc[ioff_yc[0]:ioff_yc[1]]])
    pib_tbl['yc']['suvr']['wcb'] = np.array([float(i) for i in wcb[ioff_yc[0]:ioff_yc[1]]])
    pib_tbl['yc']['suvr']['pns'] = np.array([float(i) for i in pns[ioff_yc[0]:ioff_yc[1]]])

    pib_tbl['yc']['cl'] = {}
    pib_tbl['yc']['cl']['cg']  = np.array([float(i) for i in  ccg[ioff_yc[0]:ioff_yc[1]]])
    pib_tbl['yc']['cl']['wc']  = np.array([float(i) for i in  cwc[ioff_yc[0]:ioff_yc[1]]])
    pib_tbl['yc']['cl']['wcb'] = np.array([float(i) for i in cwcb[ioff_yc[0]:ioff_yc[1]]])
    pib_tbl['yc']['cl']['pns'] = np.array([float(i) for i in cpns[ioff_yc[0]:ioff_yc[1]]])

    # > AD indices and SUVr/CL
    pib_tbl['ad']['id'] = np.array([int(i[3:]) for i in sbj[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['suvr'] = {}
    pib_tbl['ad']['suvr']['cg']  = np.array([float(i) for i in  cg[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['suvr']['wc']  = np.array([float(i) for i in  wc[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['suvr']['wcb'] = np.array([float(i) for i in wcb[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['suvr']['pns'] = np.array([float(i) for i in pns[ioff_ad[0]:ioff_ad[1]]])

    pib_tbl['ad']['cl'] = {}
    pib_tbl['ad']['cl']['cg']  = np.array([float(i) for i in  ccg[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['cl']['wc']  = np.array([float(i) for i in  cwc[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['cl']['wcb'] = np.array([float(i) for i in cwcb[ioff_ad[0]:ioff_ad[1]]])
    pib_tbl['ad']['cl']['pns'] = np.array([float(i) for i in cpns[ioff_ad[0]:ioff_ad[1]]])

    return pib_tbl
    #----------------------------------------------------------------------




def check_suvrs(suvr_yc, suvr_ad, refs):
    ''' Check/QC the obtained SUVRs in `suvr_yc` and `suvr_ad `dictionaries
        in relation to the reference in `refs` dictionary for the groups
        ('ad' or 'yc').

        Returns dictionary with the mean SUVrs and differences.
    '''

    # > prepare diff dictionary for the differences observed
    # > in young controls (yc) and AD patients (ad)
    diff = dict(yc={}, ad={})

    # > initialise the means for each reference VOI
    for rvoi in rvois:
        diff['yc'][rvoi] = dict(mean_ref=0, mean=0, N=0)
        diff['ad'][rvoi] = dict(mean_ref=0, mean=0, N=0)


    def run_checks(suvr_dct, grp):

        for rvoi in rvois:
            print('========================================================')
            for k in suvr_dct:

                if grp=='yc':
                    idx = int(k[2:5])
                elif grp=='ad':
                    idx = int(k[2:4])
                else:
                    raise ValueError('e> unknown group - only <yc>  or <ad> are accepted')
                
                i = np.where(refs[grp]['id']==idx)[0][0]

                suvr = suvr_dct[k]['suvr'][rvoi]
                suvr_ref = refs[grp]['suvr'][rvoi][i]
                err = 100*(suvr-suvr_ref)/suvr_ref

                diff[grp][rvoi]['N'] += 1
                diff[grp][rvoi]['mean_ref'] += suvr_ref
                diff[grp][rvoi]['mean'] += suvr
                diff[grp][rvoi][k] = dict(suvr=suvr, ref=suvr_ref, err=err)
                
                print(f'refvoi={rvoi}, indx> {idx}, suvr={suvr:.3f}, ref={suvr_ref:.3f}, error={err:.3f}%')

            diff[grp][rvoi]['mean'] /= diff[grp][rvoi]['N']
            diff[grp][rvoi]['mean_ref'] /= diff[grp][rvoi]['N']

            # relative % mean difference
            emean = diff[grp][rvoi]['mean']
            rmean = diff[grp][rvoi]['mean_ref']
            rmd = (emean-rmean)/rmean
            diff[grp][rvoi]['mean_diff'] = rmd
            print('--------------------------------------------------------')
            print(f'> group % mean difference: {rmd:.3f}% (suvr={emean:.3f}, ref={rmean:.3f})')

    print('========================================================')
    print('YOUNG CONTROLS (YC)')
    print('========================================================')
    run_checks(out_yc, 'yc')

    print('========================================================')
    print('AD PATIENTS (AD)')
    print('========================================================')
    run_checks(out_ad, 'ad')

    return diff