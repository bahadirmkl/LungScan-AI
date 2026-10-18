import qupath.lib.gui.scripting.QPEx
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathObject
import qupath.lib.roi.interfaces.ROI
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths

// Kaydedilecek dosyanın yolu
// Birlestirilmis COCO JSON cikti yolu (QuPath projesine gore duzenleyin)
def outputPath = buildFilePath(PROJECT_BASE_DIR, "annotations.json")

println "Kaydedilecek dosya: $outputPath"

// Proje al
def project = QPEx.getProject()
def gson = GsonTools.getInstance(true)

// Tüm görüntüler ve annotasyonlar için listeler
def allImages = []
def allAnnotations = []

// Tüm görüntüler üzerinde döngü
project.getImageList().eachWithIndex { img, index ->
    // Görüntüyü al
    def imageData = img.readImageData()
    def server = imageData.getServer()
    def hierarchy = imageData.getHierarchy()
    
    // Görüntü bilgisini ekle
    allImages << [
        id: index + 1,
        width: server.getWidth(),
        height: server.getHeight(),
        file_name: img.getImageName()
    ]

    // Görüntüdeki tüm annotasyonlar için döngü
    hierarchy.getAnnotationObjects().eachWithIndex { PathObject annotation, annIndex ->
        ROI roi = annotation.getROI()
        def points = roi.getGeometry().getCoordinates()
        def segmentation = points.collect { [it.x, it.y] }

        // ROI'nin sınırlayıcı kutusu ve alan bilgilerini al
        def bbox = [roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight()]
        def area = roi.getArea()

        // Annotasyon bilgisini ekle
        allAnnotations << [
            id: annIndex,
            iscrowd: 0,
            image_id: index + 1,
            segmentation: [segmentation.flatten()],
            bbox: bbox,
            area: area
        ]
    }
}

// Toplanan bilgileri kullanarak JSON nesnesini oluştur
def jsonMap = [
    info: [description: "my-project-name"],
    images: allImages,
    annotations: allAnnotations,
    categories: [[id: 1, name: "a"]]
]

// JSON dosyasını yaz (kaydet)
new File(outputPath).withWriter { writer ->
    gson.toJson(jsonMap, writer)
}
println "Tüm annotasyonlar kaydedildi: $outputPath"
