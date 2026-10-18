import qupath.lib.gui.scripting.QPEx
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathObject
import qupath.lib.roi.interfaces.ROI
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths

// Specify the path where the annotations will be saved
// Cikti klasoru (QuPath projesine gore duzenleyin)
def pathOutput = buildFilePath(PROJECT_BASE_DIR, "Annotations")
println "Saving path: $pathOutput"

// Create the directory if it doesn't exist
Files.createDirectories(Paths.get(pathOutput))

// Get the project
def project = QPEx.getProject()
def gson = GsonTools.getInstance(true)

// Iterate over all images in the project
project.getImageList().eachWithIndex { img, index ->
    // Retrieve the image data
    def imageData = img.readImageData()
    def server = imageData.getServer()
    def hierarchy = imageData.getHierarchy()
    def imageInfo = [
        id: index + 1,
        width: server.getWidth(),
        height: server.getHeight(),
        file_name: img.getImageName()
    ]

    // Iterate over all annotations for the image
    def annotationsList = []
    hierarchy.getAnnotationObjects().eachWithIndex { PathObject annotation, annIndex ->
        ROI roi = annotation.getROI()
        def points = roi.getGeometry().getCoordinates()
        def segmentation = points.collect { [it.x, it.y] }

        // Obtain the bounding box information for the ROI
        def bbox = [
            roi.getBoundsX(),
            roi.getBoundsY(),
            roi.getBoundsWidth(),
            roi.getBoundsHeight()
        ]
        def area = roi.getArea()

        annotationsList << [
            id: annIndex,
            iscrowd: 0,
            image_id: index + 1,
            segmentation: [segmentation.flatten()],
            bbox: bbox,
            area: area
        ]
    }

    // Create the JSON object
    def jsonMap = [
        info: [description: "my-project-name"],
        images: [imageInfo],
        annotations: annotationsList,
        categories: [[id: 1, name: "a"]]
    ]

    // Write (save) the JSON file
    def name = img.getImageName().replaceAll("\\W+", "_")
    def fileOutput = "$pathOutput/${name}.json"
    println "Saving annotation: $fileOutput"

    new File(fileOutput).withWriter { writer ->
        gson.toJson(jsonMap, writer)
    }
}
