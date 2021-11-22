
import numpy as np
#set default numpy array transform
transform_node_array = np.asarray([[-8.81943020e-01,-4.70257963e-01,3.21552180e-02,-1.93714992e+02],[ 2.64557610e-01,-4.37394327e-01, 8.59473951e-01,-1.43289179e+02],[-3.90109960e-01,7.66513960e-01,5.10167197e-01,5.78353104e+01],[ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]])

deform_node_array = np.load("D:/DATA/TEST/regis/regis_all/CMACS_15/9_to_10_array.npy")


#matrix vector multipication
def get_deform_matrix(transform_node_matrix, deformation_node_matrix):
    zs,xs,ys,_ = deformation_node_matrix.shape
    new_field = np.zeros((zs,xs,ys,3))
    count = 0
    for i_z in range(zs):
        for i_x in range(xs):
            for i_y in range(ys):
                z = np.dot(transform_node_matrix[0],np.asarray([i_z,i_x,i_y,1]))
                x = np.dot(transform_node_matrix[1], np.asarray([i_z,i_x,i_y,1]))
                y = np.dot(transform_node_matrix[2], np.asarray([i_z,i_x,i_y,1]))
                z,x,y = (int(z),int(x),int(y))
                dz = z
                if(z >= 0 and z < zs and x >= 0 and x < xs and y >= 0 and x <ys):
                    count += 1
                    print(deformation_node_matrix[z,x,y])
                    new_field[i_z,i_x,i_y] = deformation_node_matrix[z,x,y]
                else:
                    new_field[i_z,i_x,i_y] = np.zeros((3,))
    deformation_node_matrix[:,:,:,:] = new_field[:,:,:,:]
    print("count:", count)

get_deform_matrix(transform_node_array, deform_node_array)

