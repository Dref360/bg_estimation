import os
import Evaluate
import scipy
import tkMessageBox
import sys

def Utility(GT_path, evaluated_path):
    '''
    Function to evaulate the your resutls for SBMnet dataset, this code will generate a 'cm.txt' file in your result path to save all the metrics.
    input:  GT_path: the path of the groundtruth folder.
    evaluated_path: the path of your evaluated results folder.
    '''

    category_list =['backgroundMotion', 'basic', 'clutter', 'illuminationChanges', 
                        'intermittentMotion', 'jitter', 'veryLong', 'veryShort']
    category_num = len(category_list)
    result_file = os.path.join(evaluated_path, 'cm.txt')

    with open(result_file, 'w') as fid:
        fid.write('\t\tvideo\tAGE\tpEPs\tpCEPs\tMSSSIM\tPSNR\tCQM\r\n')

    m_AGE = 0
    m_pEPs = 0
    m_pCEPs = 0
    m_MSSSIM = 0
    m_PSNR = 0
    m_CQM = 0
    ipdb.set_trace()
    for category in category_list:
        print(category)
        c_AGE = 0
        c_pEPs = 0
        c_pCEPs = 0
        c_MSSSIM = 0
        c_PSNR = 0
        c_CQM = 0
        
        GT_category_path = os.path.join(GT_path, category)
        evaluated_category_path = os.path.join(evaluated_path, category)

        video_num = 0
        
        with open(result_file, 'a+') as fid:
            fid.write(category[0 : min(8, len(category))] + ': \r\n')

        for video in os.listdir(GT_category_path):

            GT_video_path = os.path.join(GT_category_path, video)
            GTs = os.listdir(os.path.join(GT_video_path))
            GT_exist = False
            MSSSIM_max = 0

            for file in GTs:
                if file.endswith('.jpg'):
                    GT_exist = True
                
                    #if more than one GT exists for the video, we keep the
                    #metrics with the highest MSSSIM value.
                    
                    GT_img = scipy.misc.imread(os.path.join(GT_video_path, file))       #background ground truth
                    evaluated_video_path = os.path.join(evaluated_category_path, video)
                    files = os.listdir(os.path.join(evaluated_video_path))       

                    for file in files:              #read the first image in the video folder
                        if file.endswith('.jpg'):
                            result_img = scipy.misc.imread(os.path.join(evaluated_video_path, file))
                            break
                    ipdb.set_trace()

                    AGE, pEPs, pCEPs, MSSSIM, PSNR, CQM = Evaluate.Evaluate(GT_img, result_img);
                    if MSSSIM > MSSSIM_max:
                        v_AGE = AGE
                        v_pEPs = pEPs
                        v_pCEPs = pCEPs
                        v_MSSSIM = MSSSIM
                        v_PSNR = PSNR
                        v_CQM = CQM
                        MSSSIM_max = MSSSIM
            if GT_exist:
                #save the video evaluation results
                with open(result_file, 'a+') as fid:
                    fid.write('\t\t' + video[0 : min(5, len(video))] + ':\t' + str(round(v_AGE, 4))+ '\t' + str(round(v_pEPs, 4)) + '\t' + str(round(v_pCEPs, 4)) + '\t' + str(round(v_MSSSIM, 4)) + '\t' + str(round(v_PSNR, 4)) + '\t' + str(round(v_CQM, 4)) + '\r\n')

                c_AGE = c_AGE + v_AGE
                c_pEPs = c_pEPs + v_pEPs
                c_pCEPs = c_pCEPs + v_pCEPs
                c_MSSSIM = c_MSSSIM + v_MSSSIM
                c_PSNR = c_PSNR + v_PSNR
                c_CQM = c_CQM + v_CQM
                video_num = video_num + 1

        c_AGE = c_AGE / float(video_num)
        c_pEPs = c_pEPs / float(video_num)
        c_pCEPs = c_pCEPs / float(video_num)
        c_MSSSIM = c_MSSSIM / float(video_num)
        c_PSNR = c_PSNR / float(video_num)
        c_CQM = c_CQM / float(video_num)

        #save the category evaluation results
        with open(result_file, 'a+') as fid:
            fid.write('\r\n' + category[0 : min(8, len(category))] + '_AVG::\t\t' + str(round(c_AGE, 4))+ '\t' + str(round(c_pEPs, 4)) + '\t' + str(round(c_pCEPs, 4)) + '\t' + str(round(c_MSSSIM, 4)) + '\t' + str(round(c_PSNR, 4)) + '\t' + str(round(c_CQM, 4)) + '\r\n\r\n')

        m_AGE = m_AGE + c_AGE
        m_pEPs = m_pEPs + c_pEPs
        m_pCEPs = m_pCEPs + c_pCEPs
        m_MSSSIM = m_MSSSIM + c_MSSSIM
        m_PSNR = m_PSNR + c_PSNR
        m_CQM = m_CQM + c_CQM

    #save the method evaluation results
    m_AGE = m_AGE / float(category_num)
    m_pEPs = m_pEPs / float(category_num)
    m_pCEPs = m_pCEPs / float(category_num)
    m_MSSSIM = m_MSSSIM / float(category_num)
    m_PSNR = m_PSNR / float(category_num)
    m_CQM = m_CQM / float(category_num)

    with open(result_file, 'a+') as fid:
        fid.write('Total:\t\t\t' + str(round(m_AGE, 4))+ '\t' + str(round(m_pEPs, 4)) + '\t' + str(round(m_pCEPs, 4)) + '\t' + str(round(m_MSSSIM, 4)) + '\t' + str(round(m_PSNR, 4)) + '\t' + str(round(m_CQM, 4)) + '\r\n')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python {0} <GT_path> <evaluated_path>".format(sys.argv[0]))

    GT_path = sys.argv[1]
    evaluated_path = sys.argv[2]
    Utility(GT_path, evaluated_path)
