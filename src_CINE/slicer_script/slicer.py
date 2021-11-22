
import numpy as np
import os
import re
import ScreenCapture
import time
import vtkSegmentationCorePython as vtkSegmentationCore

#transform matrix for 2-chamber view
default_transform_matrix = np.load(
        r"D:\DATA\CTA_CINE\eval\CT5000302027_20150625\out\regis\Reg_all\Deep_Heart_checkpoints_ML_CINE_15\deform_field\new_transform.npy")

def export_transform_node_array(transform_node_name,out_dir,file_name="transform.npy"):
    transform_node = getNode(transform_node_name)
    transform_node_array = arrayFromTransformMatrix(transform_node)
    path = os.path.join(out_dir,file_name)
    np.save(path,transform_node_array)


def screenshot(out_dir, file_name):
    cap = ScreenCapture.ScreenCaptureLogic()
    out_path = os.path.join(out_dir, file_name)
    cap.showViewControllers(False)
    cap.captureImageFromView(None, out_path)
    cap.showViewControllers(True)


# setup final deformation field
def setup_vector_display(spacing = 2.5, scale = 500, GlyphDiameterMm = 2, GlyphShaftDiameterPercent= 52,GlyphTipLengthPercent=28):
    final_field = getNode("Transform_field")
    final_field.CreateDefaultDisplayNodes()
    transformDisplayNode = final_field.GetDisplayNode()
    transformDisplayNode.SetVisibility(True)
    transformDisplayNode.SetVisibility2D(True)
    #set spacing
    transformDisplayNode.SetGlyphSpacingMm(spacing)
    transformDisplayNode.SetGlyphScalePercent(scale)
    #color map of vectors
    color_map = transformDisplayNode.GetColorMap()
    #the color of vectors are black if and only if the length of the vectors are below 0.02
    color_map.SetNodeValue(0, [0.02, 0.2, 0.2, 0.2, 0.5, 0.0])
    color_map.SetNodeValue(1, [0.02, 0.2, 0.2, 0.2, 0.5, 0.0])
    color_map.SetNodeValue(1, [0.02, 1, 1, 0, 0.5, 0.0])
    #set up red window
    red = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
    #display in 3d view
    transformDisplayNode.SetGlyphDiameterMm(GlyphDiameterMm)
    transformDisplayNode.SetGlyphShaftDiameterPercent(GlyphShaftDiameterPercent)
    transformDisplayNode.SetGlyphTipLengthPercent(GlyphTipLengthPercent)
    red = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
    transformDisplayNode.SetAndObserveRegionNode(red)


def setup(segmentation_node_name, deformation_matrix_node_name, transform_matrix, opacity = 0.6, outline = False):
    print("segmentation name:", segmentation_node_name)
    print("deformation matrix name:", deformation_matrix_node_name)
    #create transform node
    transform_node = slicer.vtkMRMLTransformNode()
    slicer.mrmlScene.AddNode(transform_node)
    transform_node.SetName("Transform")
    updateTransformMatrixFromArray(transform_node, transform_matrix)
    #apply transform to segmentation node
    segment_node = getNode(segmentation_node_name)
    segment_node.SetAndObserveTransformNodeID(transform_node.GetID())
    segment_node.SetDisplayVisibility(1)
    segment_node.GetDisplayNode().SetAllSegmentsVisibility(True)
    segment_node.CreateDefaultDisplayNodes()
    segment_display_node = segment_node.GetDisplayNode()
    segment_display_node.SetVisibility2DOutline(outline)
    segment_display_node.SetOpacity2DFill(opacity)
    # apply transform to the deformation matrix node
    deformation_matrix_node = getNode(deformation_matrix_node_name)
    deformation_matrix_node.SetAndObserveTransformNodeID(transform_node.GetID())
    #clone the transform node
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    itemIDToClone = shNode.GetItemByDataNode(transform_node)
    clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
    transform_node_copy = shNode.GetItemDataNode(clonedItemID)
    transform_node_copy.SetName("Transform_field")
    #inverse the cloned transform node
    transform_node_copy.Inverse()
    #transform_node_copy.InverseName()
    # apply deformation matrix node to transform_node_copy
    transform_node_copy.SetAndObserveTransformNodeID(deformation_matrix_node.GetID())
    setup_vector_display()
    #setup the color map
    segmentation_node = getNode(segmentation_node_name)
    segmentation = segmentation_node.GetSegmentation()
    colormap = [[0,255,0],[0,127,0],[255  ,  0 , 255],[ 255 , 255  ,  0],[  0  ,255  ,255],[   0 , 127 , 127 ],[ 255  , 95   ,95 ],[ 191  , 95  , 95  ],[  95 , 191 ,  95],[ 95  ,127  , 95],[  95  , 95  ,255], [95  , 95 , 191]]
    for i in range(12):
        segment = segmentation.GetSegment(segmentation.GetNthSegmentID(i))
        color = np.asarray(colormap[i],np.float)/255
        segment.SetColor(color)

def reset(segment_node_name):
    segment_node = getNode(segment_node_name)
    segment_node.SetDisplayVisibility(0)
    segment_node.GetDisplayNode().SetAllSegmentsVisibility(False)
    transform_node = getNode("Transform")
    transform_node_copy = getNode("Transform_field")
    slicer.mrmlScene.RemoveNode(transform_node)
    slicer.mrmlScene.RemoveNode(transform_node_copy)
    slicer.app.pythonConsole().clear()

def automate(default_transform_matrix,out_dir):
    for segment_node_name, deformation_matrix_node_name in generator(phases,deforms):
        setup(segment_node_name, deformation_matrix_node_name, default_transform_matrix)
        setup_vector_display()
        screenshot(out_dir,str(re.split(r"[._]",segment_node_name)[-3]) + ".png")
        time.sleep(5)
        reset(segment_node_name)

if __name__ == "__main__":
phases = ["phase_01.nii.gz","phase_02.nii.gz","phase_03.nii.gz","phase_04.nii.gz","phase_05.nii.gz","phase_06.nii.gz","phase_07.nii.gz","phase_08.nii.gz","phase_09.nii.gz","phase_10.nii.gz","phase_11.nii.gz","phase_12.nii.gz","phase_13.nii.gz","phase_14.nii.gz","phase_15.nii.gz","phase_16.nii.gz","phase_17.nii.gz","phase_18.nii.gz","phase_19.nii.gz","phase_20.nii.gz"]
deforms = ["deform_from_1_to_2","deform_from_2_to_3","deform_from_3_to_4","deform_from_4_to_5","deform_from_5_to_6","deform_from_6_to_7","deform_from_7_to_8","deform_from_8_to_9","deform_from_9_to_10","deform_from_10_to_11","deform_from_11_to_12","deform_from_12_to_13","deform_from_13_to_14","deform_from_14_to_15","deform_from_15_to_16","deform_from_16_to_17","deform_from_17_to_18","deform_from_18_to_19","deform_from_19_to_20","deform_from_20_to_1"]

def generator(a,b):
    assert len(a) == len(b), f"length of a: {len(a)} whereas length of b: {len(b)}"
    group = list(zip(a,b))
    for i in range(len(a)):
        yield group[i]
group = generator(phases,deforms)

names = group.__next__()
setup(names[0],names[1],default_transform_matrix)
reset(names[0])



deforms = ["deform_from_1_to_2","deform_from_2_to_3","deform_from_3_to_4","deform_from_4_to_5","deform_from_5_to_6","deform_from_6_to_7","deform_from_7_to_8","deform_from_8_to_9","deform_from_9_to_10","deform_from_10_to_11","deform_from_11_to_12","deform_from_12_to_13","deform_from_13_to_14","deform_from_14_to_15","deform_from_15_to_16","deform_from_16_to_17","deform_from_17_to_18","deform_from_18_to_19","deform_from_19_to_20","deform_from_20_to_21","deform_from_21_to_22","deform_from_22_to_23","deform_from_23_to_24","deform_from_24_to_25","deform_from_25_to_26","deform_from_26_to_27","deform_from_27_to_28","deform_from_28_to_29","deform_from_29_to_30","deform_from_30_to_31","deform_from_31_to_32","deform_from_32_to_33","deform_from_33_to_34"]

phases = ["phase_01.nii.gz","phase_02.nii.gz","phase_03.nii.gz","phase_04.nii.gz","phase_05.nii.gz","phase_06.nii.gz","phase_07.nii.gz","phase_08.nii.gz","phase_09.nii.gz","phase_10.nii.gz","phase_11.nii.gz","phase_12.nii.gz","phase_13.nii.gz","phase_14.nii.gz","phase_15.nii.gz","phase_16.nii.gz","phase_17.nii.gz","phase_18.nii.gz","phase_19.nii.gz","phase_20.nii.gz","phase_21.nii.gz","phase_22.nii.gz","phase_23.nii.gz","phase_24.nii.gz","phase_25.nii.gz","phase_26.nii.gz","phase_27.nii.gz","phase_28.nii.gz","phase_29.nii.gz","phase_30.nii.gz","phase_31.nii.gz","phase_32.nii.gz","phase_33.nii.gz"]