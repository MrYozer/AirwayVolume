import os
import numpy as np
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import SegmentStatistics

AIR_DENSITY_THRESHOLD = '-400'

#
# AirwayVolume
#

class AirwayVolume(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  # TODO: change all of these form comments

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Airway Volume" 
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# AirwayVolumeWidget
#

class AirwayVolumeWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # Reference Point List
    #
    self.ReferenceSelector = slicer.qMRMLNodeComboBox()
    self.ReferenceSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.ReferenceSelector.selectNodeUponCreation = True
    self.ReferenceSelector.addEnabled = False
    self.ReferenceSelector.removeEnabled = False
    self.ReferenceSelector.noneEnabled = False
    self.ReferenceSelector.showHidden = True
    self.ReferenceSelector.showChildNodeTypes = False
    self.ReferenceSelector.setMRMLScene( slicer.mrmlScene )
    self.ReferenceSelector.setToolTip("Set the list containing markups delineating airway sections")
    parametersFormLayout.addRow("Input Markup: ", self.ReferenceSelector)

    #
    # Orient Button
    #
    self.orientButton = qt.QPushButton("Orient")
    self.orientButton.toolTip = "Orient the volume relative to the horizontal plane created by the refence points"
    self.orientButton.enabled = False
    parametersFormLayout.addRow(self.orientButton)

    #
    # ROI selector
    #
    self.ROISelector = slicer.qMRMLNodeComboBox()
    self.ROISelector.nodeTypes = ["vtkMRMLAnnotationROINode"]
    self.ROISelector.selectNodeUponCreation = True
    self.ROISelector.addEnabled = False
    self.ROISelector.removeEnabled = False
    self.ROISelector.noneEnabled = False
    self.ROISelector.showHidden = True
    self.ROISelector.showChildNodeTypes = False
    self.ROISelector.setMRMLScene( slicer.mrmlScene )
    self.ROISelector.setToolTip( "Set the Region of Interest around the target airways")
    parametersFormLayout.addRow("Input ROI: ", self.ROISelector)

    #
    # Markup List selector
    #
    self.MarkupSelector = slicer.qMRMLNodeComboBox()
    self.MarkupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.MarkupSelector.selectNodeUponCreation = True
    self.MarkupSelector.addEnabled = False
    self.MarkupSelector.removeEnabled = False
    self.MarkupSelector.noneEnabled = False
    self.MarkupSelector.showHidden = True
    self.MarkupSelector.showChildNodeTypes = False
    self.MarkupSelector.setMRMLScene( slicer.mrmlScene )
    self.MarkupSelector.setToolTip("Set the list containing markups delineating airway sections")
    parametersFormLayout.addRow("Input Markup: ", self.MarkupSelector)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.orientButton.connect('clicked(bool)', self.onOrientButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.ROISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.MarkupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.ReferenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and \
                               self.ROISelector.currentNode() and \
                               self.MarkupSelector.currentNode() and \
                               ( self.MarkupSelector.currentNode() is not self.ReferenceSelector.currentNode())

    self.orientButton.enabled = self.inputSelector.currentNode() and \
                                self.ReferenceSelector.currentNode() and \
                                ( self.MarkupSelector.currentNode() is not self.ReferenceSelector.currentNode())

  def onOrientButton(self):
    volNode = self.inputSelector.currentNode()
    referenceList = self.ReferenceSelector.currentNode()

    logic = AirwayVolumeLogic()
    logic.autoAlign(referenceList, volNode)

  def onApplyButton(self):
    volNode = self.inputSelector.currentNode()
    ROINode = self.ROISelector.currentNode()
    markupsList = self.MarkupSelector.currentNode()
    logic = AirwayVolumeLogic()
    logic.buildAirway(volNode, ROINode, markupsList)

#
# AirwayVolumeLogic
#

class AirwayVolumeLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def buildAirway(self, inputVolume, targetROI, markupsList=None):

    volumeHandler = slicer.modules.volumes.logic()
    newVolume = volumeHandler.CloneVolume(slicer.mrmlScene, inputVolume, 'newVolume')
    volumeCropper = slicer.modules.cropvolume.logic()
    volumeCropper.CropVoxelBased(targetROI, inputVolume, newVolume)

    self.buildSegment(newVolume)

    if not markupsList is None:
      # Add the bottom coordinate of the cropped volume as a bound
      cutoffList = []
      volumeBounds = [0,0,0,0,0,0]
      newVolume.GetBounds(volumeBounds)
      for i in range(markupsList.GetNumberOfFiducials()):
        cutoffRAS = [0,0,0]
        markupsList.GetNthFiducialPosition(i, cutoffRAS)
        cutoffList.append(cutoffRAS[2])
      cutoffList.append(volumeBounds[5])

      lowerBound = volumeBounds[4]
      for idx,bound in enumerate(cutoffList):
        subVolume = volumeHandler.CloneVolume(slicer.mrmlScene, newVolume, 'subVolume')
        subROI = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLAnnotationROINode')
        centerXYZ = [0,0,0]
        targetROI.GetXYZ(centerXYZ)
        centerXYZ[2] = lowerBound + (bound - lowerBound)/2
        subROI.SetXYZ(centerXYZ)
        radiusXYZ = [0,0,0]
        targetROI.GetRadiusXYZ(radiusXYZ)
        radiusXYZ[2] = (bound - lowerBound)/2
        subROI.SetRadiusXYZ(radiusXYZ)

        volumeCropper.CropVoxelBased(subROI, newVolume, subVolume)

        self.buildSegment(subVolume, name="airway_" + str(idx))
        lowerBound = bound

        slicer.mrmlScene.RemoveNode(subVolume)

    slicer.mrmlScene.RemoveNode(newVolume)

    return True

  # FIXME: Table fails to display properly: statistics are not calculated
  def findVolume(self, segment):
    # TODO: find out whether closed surface volume or labelmap volume is more accurate and use the better one
    
    segStats = SegmentStatistics.SegmentStatisticsLogic()
    #//params = segStats.getParameterNode()
    table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    segStats.getParameterNode().SetParameter("LabelMapSegmentStatisticsPlugin.enabled", str(False))
    segStats.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", str(False))
    segStats.getParameterNode().SetParameter("Segmentation", segment)
    segStats.getParameterNode().SetParameter("MeasurementsTable", table.GetID())
    segStats.getParameterNode().SetParameter("ClosedSurfaceSegmentStatisticsPlugin.enabled", str(True))
    segStats.getParameterNode().SetParameter("visibleSegmentsOnly", str(False))
    segStats.computeStatistics()
    #//segStats.updateStatisticsForSegment(segment)
    segStats.exportToTable(table)
    segStats.showTable(table)

  # TODO: figure out how to put all of the created segments in the same segmentation
  def buildSegment(self, inputVolme, segmentationNode = None, name='airway'):
    if(segmentationNode == None):
      segmentationNode = slicer.vtkMRMLSegmentationNode()
      slicer.mrmlScene.AddNode(segmentationNode)
      segmentationNode.CreateDefaultDisplayNodes()

    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolme)
    segmentID = segmentationNode.GetSegmentation().AddEmptySegment(name)

    segmentationEditor = slicer.qMRMLSegmentEditorWidget()
    segmentationEditor.setMRMLScene(slicer.mrmlScene)
    segmentationEditor.setMRMLSegmentEditorNode(slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentEditorNode'))
    segmentationEditor.setSegmentationNode(segmentationNode)
    #//segmentationEditor.ad
    # ? What was this line supposed to be? I can't remember but it doesn't seem to be causing problems for now
    segmentationEditor.setMasterVolumeNode(inputVolme)

    segmentationEditor.setActiveEffectByName('Threshold')
    effect = segmentationEditor.activeEffect()
    effect.setParameter('MinimumThreshold', '-1024')
    effect.setParameter('MaximumThreshold', AIR_DENSITY_THRESHOLD)
    effect.self().onApply()

    segmentationEditor.setActiveEffectByName('Islands')
    effect = segmentationEditor.activeEffect()
    effect.setParameter('KeepLargestIsland', True)
    effect.self().onApply()

    segmentationNode.CreateClosedSurfaceRepresentation()
    surfaceMesh = segmentationNode.GetClosedSurfaceRepresentation(segmentID)
    normals = vtk.vtkPolyDataNormals()
    normals.ConsistencyOn()
    normals.SetInputData(surfaceMesh)
    normals.Update()
    surfaceMesh = normals.GetOutput() 

  def autoAlign(self, markupsList, inputVolume):
    coordinateList = []
    for i in range(markupsList.GetNumberOfFiducials()):
      x = [0,0,0]
      markupsList.GetNthFiducialPosition(i, x)
      coordinateList.append(x)

    # TODO: throw an exception if there are not exactly three points in the list
    
    vectorA = [coordinateList[1][dim] - coordinateList[0][dim] for dim in range(3)]
    vectorB = [coordinateList[2][dim] - coordinateList[0][dim] for dim in range(3)]

    # calculate the normal vector to the plane and dvide it by its length to create a unit vector
    vectorNorm = np.cross(vectorA, vectorB)
    length = np.sum(np.power(dim, 2) for dim in vectorNorm)
    length = np.sqrt(length)
    vectorNorm /= length
    
    # weird fact: RAS coordinates work like z, x, y if you think from a lateral view
    v = np.cross(vectorNorm, [0,0,1])
    # v represents the skew-symmetric cross product of the normal vector and the vertical unit vector
    v = [[     0, -v[2],  v[1]],
         [  v[2],     0, -v[0]],
         [ -v[1],  v[0],     0]]

    x = [[ 1, 0, 0],
         [ 0, 1, 0],
         [ 0, 0, 1]]

    n = np.add(x, v)
    # c represents the cosine of the angle between the given normal and the vertical unit vector
    c = (vectorNorm[2] / np.sqrt(np.sum(np.power(dim, 2) for dim in vectorNorm)))
    logging.info('arccos')
    logging.info(np.arccos(c))
    m = np.linalg.matrix_power(v, 2) * (1/(1+c))

    r = np.add(n, m)

    # correction matrix: for some reason the computed transform is mirrored in the x and z axes,
    # so a correction matrix is used to rectify it
    x = [[ 1, 0, 0],
         [ 0,-1, 0],
         [ 0, 0,-1]]
    r = np.dot(r,x)

    ## Offsets x rotation based on the centered position of the first point 
    #xOffset = coordinateList[0][0]
    #yOffset = coordinateList[0][1]
    #tOffset = np.arcsin(xOffset/yOffset)
    #x = [[ np.cos(tOffset), -np.sin(tOffset), 0],
         #[ np.sin(tOffset), np.cos(tOffset), 0],
         #[               0,               0, 0]]
    #r = np.dot(r,x)

    r2 = np.c_[ r, np.zeros(3)]
    rotMatrix = np.r_[ r2, [[0,0,0,1]]]

    vtkRotMatrix = vtk.vtkMatrix4x4()
    [ [ vtkRotMatrix.SetElement(row, col, rotMatrix[row, col]) for row in range(4) ] for col in range(4) ]
    transform = slicer.vtkMRMLTransformNode()
    transform.SetMatrixTransformToParent(vtkRotMatrix)
    slicer.mrmlScene.AddNode(transform)
    inputVolume.SetAndObserveTransformNodeID(transform.GetID())

class AirwayVolumeTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_AirwayVolume1()

  def test_AirwayVolume1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = AirwayVolumeLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
