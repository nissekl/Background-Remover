from numpy import *
import numpy
import maxflow
from PIL import Image
from matplotlib import pyplot as plt
from pylab import *
import cv2
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sys import argv 

'''Setting Global Definition '''
drawing = False # true if mouse is pressed
fg_draw = False # true if you are drawing strokes for foreground
bg_draw = False # true if you are drawing strokes for background
fg_pos = [] #save foreground stroked position
bg_pos = [] #save background stroked position 
Save_Id = 1



def draw_stroke(event,x,y,flags,param):
    global drawing, img_for_draw 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if fg_draw and [x, y] not in fg_pos: fg_pos.append([x, y])
        elif bg_draw and [x, y] not in bg_pos: bg_pos.append([x, y])
        cv2.circle(img_for_draw,(x,y),2,(bg_draw*225,0,fg_draw*255),-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
           if fg_draw and [x, y] not in fg_pos: fg_pos.append([x, y])
           elif bg_draw and [x, y] not in bg_pos: bg_pos.append([x, y])
           cv2.circle(img_for_draw,(x,y),2,(bg_draw*225,0,fg_draw*255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False



def graph_cut(fore, back, image, cluster_num):
    m,n = image.shape[0],image.shape[1]
    
    
    '''Coverting pixels to a N*3 matrix, 3 is the color channel'''  
    fore_color_val = np.zeros((len(fore),3))
    back_color_val = np.zeros((len(back),3))

    
    #Get color values
    for i, j in enumerate(fore):
        fore_color_val[i, :] = image[j[1],j[0],:] #Careful, the value obtained from draw_stroke is [col, row], so need to be opposite 
    
    for i, j in enumerate(back):
        back_color_val[i, :] = image[j[1],j[0],:] #Careful, the value obtained from draw_stroke is [col, row], so need to be opposite 


    '''Do the K means and calculate the mean color value of each cluster'''
    #K-Means from Sklearn
    K = cluster_num
    kmean_cluster_fore = cluster.KMeans(n_clusters=K)
    kmean_cluster_fore.fit(fore_color_val)
    cluster_fore_label = kmean_cluster_fore.labels_
    
    kmean_cluster_back = cluster.KMeans(n_clusters=K)
    kmean_cluster_back.fit(back_color_val)
    cluster_back_label = kmean_cluster_back.labels_
    
    mean_fore, var_fore = get_means_var(fore_color_val, cluster_fore_label, K)
    mean_back, var_back = get_means_var(back_color_val, cluster_back_label, K)


    '''Use the mean color value from each cluster to calculate the probability of each pixel'''
    F,B = ones((image.shape[0],image.shape[1])),np.ones((image.shape[0],image.shape[1]))

    for i in range(image.shape[0]):
       for j in range(image.shape[1]):
           dis_f = np.linalg.norm(mean_fore-image[i,j,:], axis=1)**2
           var_f = var_fore**2
           dis_b = np.linalg.norm(mean_back-image[i,j,:], axis=1)**2
           var_b = var_back**2
           f = np.min(dis_f/var_f)
           b = np.min(dis_b/var_b)
           F[i,j] = exp(-f)
           B[i,j] = exp(-b)
    
    #plot_prob_map(F, B, image)
    F,B = F.reshape(-1,1),B.reshape(-1,1)


    """MaxFlow Structure Setting"""
    g = maxflow.Graph[float](m,n) # define the graph    
    nodes = g.add_nodes(m*n) # Adding non-nodes
    s=1
    Im = image.reshape(m*n,3)
    for i in range(m*n):#checking the 4-neighborhood pixels
        ws=(F[i]/(F[i]+B[i])) # source weight
        wt=(B[i]/(F[i]+B[i])) # sink weight
       
        g.add_tedge(i,ws,wt) # edges between pixels and terminal
        if i%n != 0: # for left pixels
            w = exp(-0.5*(linalg.norm(Im[i]-Im[i-1])/s)**2) # the cost function for two pixels
            g.add_edge(i,i-1,w,w) # edges between two pixels
        
        if (i+1)%n != 0: # for right pixels
            w = exp(-0.5*(linalg.norm(Im[i]-Im[i+1])/s)**2)
            g.add_edge(i,i+1,w,w) # edges between two pixels
        
        if i//n != 0: # for top pixels
            w = exp(-0.5*(linalg.norm(Im[i]-Im[i-n])/s)**2)
            g.add_edge(i,i-n,w,w) # edges between two pixels
        
        if i//n != m-1: # for bottom pixels
            w = exp(-0.5*(linalg.norm(Im[i,:]-Im[i+n,:])/s)**2)
            g.add_edge(i,i+n,w,w) # edges between two pixels
    
    flow = g.maxflow()#Remember to cut after assigning weights!!!!!
    
    Iout = ones(shape = nodes.shape)
    for i in range(len(nodes)):
        Iout[i] = g.get_segment(nodes[i]) # calssifying each pixel as either forground or background    
    

    '''Seperate the fore and back'''
    Iout=Iout.reshape((m,n))# reshape back to image size to assgin to background and foreground
    fore_img = zeros((m,n,3)) 
    back_img = zeros((m,n,3))

    for i in range(m):
        for j in range(n): # converting the True/False to Pixel intensity
            if Iout[i,j]==False:
                fore_img[i,j,:] = image[i,j,:] # foreground for 3d image
            else: back_img[i,j,:] = image[i,j,:] #background for 3d image
    
    return fore_img, back_img



def get_means_var(stroked_pts, stroked_labels, K):
    cluster_means = zeros((K,3))
    cluster_val= zeros((K,1))
    for i in range(K):
        pts = stroked_pts[argwhere(stroked_labels==i)[:,0],:]
        cluster_means[i,:] = mean(pts,axis=0)
        cluster_val[i,0] = var(pts)
    return cluster_means, cluster_val



def save_img(fore_img, back_img, path_name):
    global Save_Id
    '''Setting Saving Pathname'''
    fore_name = path_name.split('.')[0]+'_fore_%d' % Save_Id +'.png'
    fore_trans_name = path_name.split('.')[0]+'_transfore_%d' % Save_Id +'.png'
    back_name = path_name.split('.')[0]+'_back_%d' % Save_Id +'.png'

    #Process transparent image
    fore_img_trans = np.float32(fore_img)
    tmp = cv2.cvtColor(fore_img_trans, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(fore_img_trans)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    
    #Save file
    cv2.imwrite(fore_name, fore_img)
    cv2.imwrite(fore_trans_name, dst)
    cv2.imwrite(back_name, back_img)
    Save_Id+=1
    print("Imgs Saved")



def draw_fgimg():
    global fg_draw, bg_draw, img_for_draw
    
    '''Draw the Foreground'''
    cache1 = img_for_draw.copy()  # copy an img for refresh and grah_cut calculation
    fg_draw = True
    cv2.namedWindow('Draw Foreground and Press n to next step')
    cv2.setMouseCallback('Draw Foreground and Press n to next step', draw_stroke)
    while True:
        cv2.imshow('Draw Foreground and Press n to next step', img_for_draw)
        k = cv2.waitKey(33)
        if k == ord('n'): #press n to end drawing foreground 
            fg_draw = False
            break 
        elif k == ord('d'): # press d to delete the current drawing
            img_for_draw = cache1.copy()
            fg_pos.clear()
            print('Clear the strokes, you can draw again')
    cache2 = img_for_draw.copy()
    cv2.destroyWindow('Draw Foreground and Press n to next step')


    '''Draw the Background'''
    bg_draw = True
    cv2.namedWindow('Draw Background and Press n to next step')
    cv2.setMouseCallback('Draw Background and Press n to next step', draw_stroke)
    while True:
        cv2.imshow('Draw Background and Press n to next step', img_for_draw)   
        k = cv2.waitKey(33)    
        if k == ord('n'): #press n to end drawing foreground 
            bg_draw = False
            break 
        elif k == ord('d'): # press d to delete the current drawing
            img_for_draw = cache2.copy()
            bg_pos.clear()
            print('Clear the strokes, you can draw again')
    cv2.destroyWindow('Draw Background and Press n to next step')



'''Import Image'''
img_path = argv[1]
img = cv2.imread(img_path)
img_for_draw = img.copy()#this is for drawing line display


'''Draw foreground and background'''
draw_fgimg()
cluster_num = input('Enter the Cluster Factor:')
fore_img, back_img = graph_cut(fg_pos, bg_pos, img, int(cluster_num))
img_for_display = np.hstack((fore_img,back_img))


while True:
    cv2.imshow('res',img_for_display/255)
    k = cv2.waitKey(33)   
    if k == ord('e'):break #press e to close the app
    if k == ord('r'): #press r to redraw and recalculate the photo
        fg_pos.clear()
        bg_pos.clear()
        img_for_draw = img.copy()
        draw_fgimg()
        cluster_num = input('Enter the Cluster Factor:')
        fore_img, back_img = graph_cut(fg_pos, bg_pos, img, int(cluster_num))
        img_for_display = np.hstack((fore_img,back_img))
    if k == ord('s'): #press s to save the img
        save_img(fore_img, back_img, img_path)