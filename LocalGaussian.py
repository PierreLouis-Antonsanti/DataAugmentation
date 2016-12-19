#!/usr/bin/env python
# -*- coding: utf-8 -*-

# /Users/piant/PycharmProjects/DatAug/LocalGaussian.py
# Local Gaussian Approximation


""" !!! A FAIRE !!! :
- La classe fille multiscale :
    - Créer dans self les Luk = uk * G
    - Créer dans self les zoom out
- TextureSynthesis compile mais ne fonctionne pas !
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.fftpack
import skimage.graph
from RemplissageRegion import *

class ValidGaussian(object):

    def __init__(self,texture_sample,patch_size = 30, n_neighbors = 10, overlap_rate = 0.5):

        """ Input :
                   texture sample : the image of the texture sample, size [M,N,n_shape],
                   patch_size : size of the needed patch side,
                   n_neighbors : number of nearest neighbors."""

        # Parameters for the gaussian approximation
        self.texture_sample = texture_sample
        self.patch_size = patch_size
        self.n_neighbors = n_neighbors
        self.dim_texture = np.asarray(np.shape(texture_sample))
        self.overlap_size = int(overlap_rate*patch_size)

        if len(self.dim_texture) == 2:
            self.dim_texture = np.concatenate((self.dim_texture,[1]))
            self.texture_sample = np.asarray(list(self.texture_sample)).reshape(self.dim_texture)

        self.dim_dist = self.patch_size * self.patch_size * self.dim_texture[2]


    def Patch(self, coordinates = [0,0]):

        """ Input : 
                   texture sample : the image of the texture sample, size [M,N,n_shape],  
                   patch_size : size of the needed patch side,
                   coordinates : the coordinates of the patch (basically the top left corner).
            Output :
                   patch : the extracted patch of size [n*n,1]  """

        
        if self.patch_size > self.dim_texture[0] or self.patch_size > self.dim_texture[1]:
            return "The patch size is too big."
        else:
            if coordinates[0]+self.patch_size > self.dim_texture[0] or coordinates[1]+self.patch_size > self.dim_texture[1]:
                return "The patch is out of the picture"
            else:
                patch = np.asarray(self.texture_sample[coordinates[0]:coordinates[0]+self.patch_size,
                                                       coordinates[1]:coordinates[1]+self.patch_size])
                
                return patch


    def GeneratePatchSet(self):

        """Generate the patches and store them in a [n_shape*patch_size*patch_size,n_patches] matrix"""

        self.PatchSet = np.zeros([self.dim_dist, (self.dim_texture[0] - self.patch_size)
                                  * (self.dim_texture[1] - self.patch_size)])

        cpt = 0
        for x in range(self.dim_texture[0] - self.patch_size):
            for y in range(self.dim_texture[1] - self.patch_size):
                patch = self.Patch([x, y])
                self.PatchSet[:, cpt] = np.reshape(patch,(self.dim_dist),order = 'F')
                cpt += 1


    def NearestPatches(self, initial_patch):

        """ Input :
                   initial_patch : size [n*n,1] : the patch we look for the neighbors,
            Output :
                   neighbors_set: [n*n*n_shape,n_neighbors] : n_neighbors nearest patches.
        """

        neighbors_set = np.zeros([self.dim_dist,self.n_neighbors])
        
        initial_patch = np.reshape(initial_patch,(self.dim_dist,1),order = 'F')
        diff = self.PatchSet - initial_patch
        distances = np.sum(diff * diff,axis=0)

        for k in range(self.n_neighbors):
            patch_index = np.argmin(distances)
            patch = self.PatchSet[:,patch_index]
            distances[patch_index] = np.inf
            neighbors_set[:,k] = patch
        
        return neighbors_set



    def LocalGaussianEstimation(self,neighbors_set):

        """
            Input : neighbors_set: [n*n*n_shape,n_neighbors] : n_neighbors nearest patches.
            Output : estimated_patch : [n*n*n_shape,1] : the patch following Local Multivariate Gaussian Distribution.
        """

        local_mean = np.mean(neighbors_set,axis=1).reshape((neighbors_set.shape[0],1),order = 'F')

        P = neighbors_set-local_mean
        
        if self.n_neighbors > 1:

            local_covariance = (1/self.n_neighbors-1)*P.dot(P.transpose())
            """
            eigenval,W = np.linalg.eig(P.transpose().dot(P))
            D = np.diag(eigenval)

            p_est = np.random.multivariate_normal(np.zeros(self.n_neighbors), np.eye(self.n_neighbors),[1])
            p_est = np.transpose(p_est)

            estimated_patch = (1/self.n_neighbors-1)*P.dot(W.dot(D.dot(p_est))) + local_mean
            """
            estimated_patch = np.random.multivariate_normal(local_mean[:,0], local_covariance,[1])
            estimated_patch = np.transpose(estimated_patch)

            estimated_patch = np.reshape(estimated_patch, (self.patch_size,self.patch_size,self.dim_texture[2])
                                                                                                           ,order = 'F')
        else:
            local_covariance = 0.01
            estimated_patch = neighbors_set
            np.reshape(estimated_patch, (self.patch_size,self.patch_size,self.dim_texture[2]),order = 'F')

        return estimated_patch
            
    
    def LocalSynthesize(self,coordinates = [0,0]):

        """ Input : coordinates : the coordinates of the patch (basically the top left corner).
            Output : u_tild : the synthesized texture  """

        self.GeneratePatchSet()
        [x,y] = coordinates
        u_tild = np.asarray(list(self.texture_sample))

        initial_patch = self.Patch(coordinates)
        neighbors_set = self.NearestPatches(initial_patch)
        estimated_patch = self.LocalGaussianEstimation(neighbors_set)
        u_tild[x:x+self.patch_size,y:y+self.patch_size] =  estimated_patch

        if u_tild.shape[2] == 1:
            u_tild = u_tild[:,:,0]

        return u_tild


#######################################################################################################


class TextureSynthesis(ValidGaussian):

    def __init__(self,texture_sample,patch_size = 30, n_neighbors = 10, overlap_rate = 0.5, ratio = 2):

        """ Input :
                   texture sample : the image of the texture sample, size [M,N,n_shape],
                   patch_size : size of the needed patch side,
                   n_neighbors : number of nearest neighbors.
        """


        ValidGaussian.__init__(self, texture_sample, patch_size, n_neighbors, overlap_rate)
        self.ratio = ratio
        self.synthesised_texture = np.zeros([int(ratio*self.dim_texture[0]),int(ratio*self.dim_texture[1]),
                                                                                                  self.dim_texture[2]])

        # Projectors for the overlap :

        self.VOP = np.zeros((self.dim_dist,1))
        cpt = 0
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                if x <= self.overlap_size:
                    self.VOP[cpt] += 1
                cpt += 1

        self.HOP = np.zeros((self.dim_dist, 1))
        cpt = 0
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                if y <= self.overlap_size:
                    self.HOP[cpt] += 1
                cpt += 1


        self.LOP = np.zeros((self.dim_dist, 1))
        cpt = 0
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                if x <= self.overlap_size or y <= self.overlap_size:
                    self.LOP[cpt] += 1
                cpt += 1

    def NearestOverlappingPatches(self, coordinates, overlap_shape):
        """ Input :
                   coordinates : the coordinates of the current synthesised patch,
                   overlap_shape : string : 'L' : L-shaped  ,'V' : vertical shaped, 'H' : horizontal shaped.
            Output :
                   neighbors_set: [n*n*n_shape,n_neighbors] : n_neighbors nearest patches.
        """
        [x, y] = coordinates

        if x<self.dim_texture[0]-self.patch_size and y<self.dim_texture[1]-self.patch_size:
            initial_patch = self.Patch(coordinates)
        else:
            initial_patch = self.synthesised_texture[x:x+self.patch_size,y:y+self.patch_size]

        initial_patch = np.reshape(initial_patch, (self.dim_dist, 1), order='F')

        if overlap_shape == 'L':
            proj = self.LOP
        elif overlap_shape == 'V':
            proj = self.VOP
        elif overlap_shape == 'H':
            proj = self.HOP
        else:
            return 'Wrong shape : use string: ''L'' : L-shaped  ,''V'' : vertically shaped, ''H'' : horizontally shaped.'

        initial_patch = proj[:,0]*initial_patch[:]
        neighbors_set = np.zeros([self.dim_dist, self.n_neighbors])
        diff = np.zeros(self.PatchSet.shape)


        for i in range(self.PatchSet.shape[1]):
            # print((proj[:,0] * self.PatchSet[:, i]).shape)
            # print(initial_patch.shape)
            diff[:, i] = proj[:,0] * self.PatchSet[:, i]-initial_patch[:,0]

        distances = np.sum(diff * diff, axis=0)

        for k in range(self.n_neighbors):
            patch_index = np.argmin(distances)
            patch = self.PatchSet[:, patch_index]
            distances[patch_index] = np.inf
            neighbors_set[:, k] = patch
            """plt.figure(2)
            plt.imshow(np.reshape(neighbors_set[:, k], [self.patch_size, self.patch_size], order='F'), cmap='gray')
            plt.show()"""
        return neighbors_set

    def Boundary(self,current_patch,coordinates,overlap_shape):
        """
        Input :
            coordinates : the coordinates of the current synthesised patch,
            overlap_shape : string : 'L' : L-shaped  ,'V' : vertical shaped, 'H' : horizontal shaped.
        Output :
            patch_area_conserved: [patch_size*patch_size] : the area of the patch we'll quilt, using the shortest path in the overlapping area.
        """
        [x,y] = coordinates

        if overlap_shape == 'L':
            proj = self.LOP
        elif overlap_shape == 'V':
            proj = self.VOP
        elif overlap_shape == 'H':
            proj = self.HOP
        else:
            return 'Wrong shape :use string: ''L'' : L-shaped  ,''V'' : vertically shaped, ''H'' : horizontally shaped.'

        current_patch = np.reshape(current_patch, (self.dim_dist, 1),order = 'F')
        area = (proj*current_patch).reshape([self.patch_size,self.patch_size,self.dim_texture[2]],order = 'F')

        map = self.synthesised_texture[x:x+self.patch_size,y:y+self.patch_size]-area
        map = np.sum(map*map,axis = 2)
        map[map == 0] = np.inf

        # plt.imshow(map)
        # plt.show()
        sys.setrecursionlimit(self.patch_size*self.patch_size)

        if overlap_shape == 'V':

            start = np.argmin(map[0,:])
            end = np.argmin(map[self.patch_size-1,:])
            indices, coast = skimage.graph.route_through_array(map, (0, start),(self.patch_size - 1, end))
            indices = np.array(indices).T
            bound_path = np.zeros_like(map)
            bound_path[indices[0], indices[1]] = 1

            # Create a filter with ones on the area of the patch we conserve, 0 anywhere else.
            patch_area_conserved = np.asarray(list(bound_path))
            for indx in range(len(bound_path[:,0])):
                if patch_area_conserved[indx,0]==0:
                    patch_area_conserved = fill(patch_area_conserved, self.patch_size, self.patch_size, indx, 0)

            patch_area_conserved = 1 - patch_area_conserved

        elif overlap_shape == 'H':

            start = np.argmin(map[:,0])
            end = np.argmin(map[:,self.patch_size-1])
            indices, coast = skimage.graph.route_through_array(map, (start,0),(end, self.patch_size - 1))
            indices = np.array(indices).T
            bound_path = np.zeros_like(map)
            bound_path[indices[0], indices[1]] = 1

            # Create a filter with ones on the area of the patch we conserve, 0 anywhere else.
            patch_area_conserved = np.asarray(list(bound_path))
            for indy in range(len(bound_path[0,:])):
                if patch_area_conserved[0,indy]==0:
                    patch_area_conserved = fill(patch_area_conserved, self.patch_size, self.patch_size, 0, indy)
            patch_area_conserved = 1 - patch_area_conserved

        elif overlap_shape == 'L':

            start = np.argmin(map[self.patch_size - 1, :])
            end = np.argmin(map[:, self.patch_size - 1])
            indices, coast = skimage.graph.route_through_array(map, (self.patch_size - 1, start),
                                                               (end, self.patch_size - 1))
            indices = np.array(indices).T
            bound_path = np.zeros_like(map)
            bound_path[indices[0], indices[1]] = 1

            # Create a filter with ones on the area of the patch we conserve, 0 anywhere else.
            patch_area_conserved = np.asarray(list(bound_path))
            for indx in range(len(bound_path[:,0])):
                if patch_area_conserved[indx,0]==0:
                    patch_area_conserved = fill(patch_area_conserved, self.patch_size, self.patch_size, indx, 0)
            for indy in range(len(bound_path[0, :])):
                if patch_area_conserved[0, indy] == 0:
                    patch_area_conserved = fill(patch_area_conserved, self.patch_size, self.patch_size, 0, indy)
            patch_area_conserved = 1 - patch_area_conserved

        """plt.ion()
        plt.figure()
        plt.subplot(221)
        plt.imshow(map)
        plt.subplot(222)
        plt.imshow(bound_path, cmap='gray')
        plt.subplot(223)
        plt.imshow(patch_area_conserved, cmap='gray')
        plt.show()
        plt.pause(0.1)
        plt.close()"""

        return patch_area_conserved


    def Synthesis(self, seed_coordinates=[0, 0]):

        """ Input : coordinates : the coordinates of the patch (basically the top left corner).
            Output : u_tild : the synthesized texture  """

        self.GeneratePatchSet()
        [x, y] = seed_coordinates

        # Generate the first patch:
        seed_patch = self.Patch(seed_coordinates)
        neighbors_set = self.NearestPatches(seed_patch)
        estimated_patch = self.LocalGaussianEstimation(neighbors_set)

        self.synthesised_texture[0:self.patch_size, 0:self.patch_size] = estimated_patch

        # Generate the other patches:
        for x in np.arange(0, int(self.ratio*self.dim_texture[0])-self.patch_size, self.patch_size-self.overlap_size):
            for y in np.arange(0, int(self.ratio*self.dim_texture[1])-self.patch_size, self.patch_size-self.overlap_size):

                if x > 0 or y > 0:
                    # Compute the overlap shape :
                    if x == 0:
                        overlap_shape = 'V'
                    elif y == 0:
                        overlap_shape = 'H'
                    else:
                        overlap_shape = 'L'

                    print(x,y)
                    # print(overlap_shape)

                    neighbors_set = self.NearestOverlappingPatches([x,y], overlap_shape)
                    estimated_patch = self.LocalGaussianEstimation(neighbors_set)
                    patch_area_conserved = self.Boundary(estimated_patch,[x,y],overlap_shape)

                    """plt.figure(1)
                    plt.subplot(223)
                    plt.imshow((1-patch_area_conserved)*self.synthesised_texture[x:x + self.patch_size, y:y + self.patch_size,0] , cmap='gray')
                    plt.subplot(224)
                    plt.imshow(patch_area_conserved*estimated_patch[:,:,0], cmap='gray')
                    plt.subplot(221)
                    plt.imshow(self.synthesised_texture[x:x + self.patch_size, y:y + self.patch_size,0] , cmap='gray')
                    plt.subplot(222)
                    plt.imshow(estimated_patch[:,:,0], cmap='gray')
                    plt.show()"""

                    for layer in range(self.dim_texture[2]):
                        self.synthesised_texture[x:x + self.patch_size, y:y + self.patch_size,layer] = \
                            (1-patch_area_conserved)*self.synthesised_texture[x:x + self.patch_size, y:y + self.patch_size,layer] \
                            + patch_area_conserved*estimated_patch[:,:,layer]


        if self.synthesised_texture.shape[2] == 1:
            self.synthesised_texture = self.synthesised_texture[:, :, 0]

        return self.synthesised_texture


##############################################################################################################

class MSLG(TextureSynthesis):
    """Multi Scale Local Gaussian"""

    def __init__(self, texture_sample, patch_size=30, n_neighbors=10, overlap_rate=0.5, ratio=2, scale_number = 3):
        """ Input :
                   texture sample : the image of the texture sample, size [M,N,n_shape],
                   patch_size : size of the needed patch side,
                   n_neighbors : number of nearest neighbors."""

        ValidGaussian.__init__(self, texture_sample, patch_size, n_neighbors, overlap_rate)
        TextureSynthesis.__init__(self,self,texture_sample,patch_size,n_neighbors, overlap_rate, ratio)
        self.scale_number = scale_number


    def zoom_out(self,image):

        img = scipy.ndimage.filters.gaussian_filter(image, sigma=1.4)
        img_zoomed_out = np.zeros((int(self.dim_texture[0]/2),int(self.dim_texture[1]/2),self.dim_texture[2]))

        for i in range(int(self.dim_texture[0]/2)):
            for j in range(int(self.dim_texture[1]/2)):
                img_zoomed_out[i,j] = img[2*i,2*j]

        return img_zoomed_out


    def zoom_in(self, image):

        u_chap = scipy.fftpack.fftn(image,shape = self.dim_texture, axis = [0,1])
        img_zoomed_in = np.fft.ifftn(u_chap, shape = [self.dim_texture[0]*2, self.dim_texture[1]*2,self.dim_texture[2]],
                                                                                                     axis = [0,1]).real
        return img_zoomed_in

    def multiscaled_distance(self,patch1, patch2,Luk,vk,proj):
        """
        Input :
            patch1 : ,
            patch2 : ,
            overlap_shape : string : 'L' : L-shaped  ,'V' : vertical shaped, 'H' : horizontal shaped.
        Output :
            patch_area_conserved: [patch_size*patch_size] : the area of the patch we'll quilt, using the shortest path in the overlapping area.
        """
        n = self.patch_size

        dist = 1/(np.sum(proj[:,0]))*np.sum((proj[:,0]*patch1-proj[:,0]*patch2)**2) +\
                                         1/(self.patch_size**2)*np.sum((Luk-vk)**2)

        return dist

    def Multiscale_NOP(self, coordinates, overlap_shape, rang):
        """ Input :
                   coordinates : the coordinates of the current synthesised patch,
                   overlap_shape : string : 'L' : L-shaped  ,'V' : vertical shaped, 'H' : horizontal shaped.
            Output :
                   neighbors_set: [n*n*n_shape,n_neighbors] : n_neighbors nearest patches.
        """
        [x, y] = coordinates


        if x<self.dim_texture[0]-self.patch_size and y<self.dim_texture[1]-self.patch_size:
            initial_patch = self.Patch(coordinates)
        else:
            initial_patch = self.synthesised_texture[x:x+self.patch_size,y:y+self.patch_size]

        initial_patch = np.reshape(initial_patch, (self.dim_dist, 1), order='F')

        if overlap_shape == 'L':
            proj = self.LOP
        elif overlap_shape == 'V':
            proj = self.VOP
        elif overlap_shape == 'H':
            proj = self.HOP
        else:
            return 'Wrong shape : use string: ''L'' : L-shaped  ,''V'' : vertically shaped, ''H'' : horizontally shaped.'

        neighbors_set = np.zeros([self.dim_dist, self.n_neighbors])
        distances = np.zeros([self.PatchSet.shape[1],1])+np.inf

        for i in range(self.PatchSet.shape[1]):
            # print((proj[:,0] * self.PatchSet[:, i]).shape)
            # print(initial_patch.shape)
            distances[i] = self.multiscaled_distance(self,initial_patch[:, 0],self.PatchSet[:, i],
                                                                      self.LukSet[:,i,rang],self.vk[:, i],proj)

        for k in range(self.n_neighbors):
            patch_index = np.argmin(distances)
            patch = self.PatchSet[:, patch_index]
            distances[patch_index] = np.inf
            neighbors_set[:, k] = patch

        return neighbors_set

    def Multiscale_texture_Synthesis(self,seed_coordinates=[0, 0]):

        """ Input : coordinates : the coordinates of the patch (basically the top left corner).
            Output : u_tild : the synthesized texture  """

        uk = [self.texture_sample]
        self.LukSet = np.zeros((np.concatenate((self.dim_texture,[self.scale_number-1]))))

        for k in range(self.scale_number-1):
            uk.append(self.zoom_out(uk[k]))
            self.LukSet[:,:,:,k] = scipy.ndimage.filters.gaussian_filter(uk[k], sigma=1.4)

        wk = [self.Synthesis(seed_coordinates)]

        for k in np.arange(self.scale_number-2,0,self.scale_number-1):
            vk = self.zoom_in(wk[k])

