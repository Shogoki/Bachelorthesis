\clearpage

# Umsetzung \label{chapter_umsetzung}

In diesem Teil wird die Umsetzung der Arbeit beschrieben. Im ersten Abschnitt wird auf die Art und Einteilung der zu klassifizierenden Daten eingegangen. Im weiteren Verlauf wird die Beschaffung der verschiedenen Datensätze behandelt, mit welchen gearbeitet werden soll. Anschließend werden diverse Verfahren der Datenpräparation wie Vorbereitung, Normalisierung und Datenvervielfältigung thematisiert. Letzten Endes werden verschiedene neuronale Netze entworfen, trainiert und getestet, bis dann im letzten Abschnitt die Realisierung des zu erstellenden Webservice näher betrachtet wird.

## Beschaffenheit und Einteilung der Daten

Wie bereits beschrieben, sollen für den späteren Webservice Videodaten als Eingabe dienen. Das neuronale Netz wird jedoch nicht direkt die Videodaten, sondern aus dem Video extrahierte Einzelbilder als Eingabe erhalten. Diese Eingabedaten werden anhand später beschriebener Verfahren noch weiter vorbereitet und optimiert.
Die Einzelbilder sollen als *Array* von Pixelwerten eines Graustufenbildes an das neuronale Netzwerk übergeben werden, siehe Kapitel Datenpräparation \ref{chapter_dataprep}.
Die Bilder sollen in 8 verschiedene Klassen eingeteilt werden. Die verschiedenen Emotionen des *FACS* und die zusätzliche Klasse *neutral* bilden den Zielklassenvektor $Z$, es gilt also $|Z| = 8$. Jedes Bild, welches der selben Emotion des *FACS* entspricht, ist Mitglied der selben Klasse aus $Z$.

## Datensätze

Ein Teil der Arbeit bestand darin, geeignete Datensätze für das Training und die Evaluierung des neuronalen Netzes zu finden und diese später in entsprechende Trainings, Evaluierungs und Test-Datensätze für das neuronale Netzwerk zu unterteilen.

### Beschaffung der Datensätze

Dazu sollen 2 verschiedene Ansätze unterschieden werden. Zum einen die Beschaffung eines vorhandenen freien Datensatzes von Gesichtsbildern inklusive der Zuordnung zu einer der entsprechenden *FACS* Emotionen, sowie zum anderen die Generierung von eigenen Daten mithilfe eines Webservice und freiwillgen Probanden. 
In dieser Arbeit wurden beide Ansätze in Kombination verwendet. Es wurden also 2 verschiedene Datenquellen herangezogen, was bei der Aufteilung der Datensätze eine wichtige Rolle spielt (siehe Kapitel Einteilung der Datensätze \ref{chapter_datasplit}). 

#### FER+

Als großer frei Verfügbarer Datensatz wurde der *Facial Expression Recognition+* (FER+)[@Barsoum2016] Datensatz verwendet. Bei den Eingangsdaten des *FER+* handelt es sich um dieselben Bilder, wie auch beim *FER2013*, welcher Teil der International Conference for Machine Learning (ICML) Challenge 2013 war und danach der Öffentlichkeit zur Verfügung gestellt wurde. Bei FER+ wurden jedoch alle *Label* mithilfe von *Crowdsourcing* neu erstellt, um eine bessere Datenqualität zu erreichen (vgl. [@Barsoum2016]). Der Datensatz besteht aus 34034 48x48 Graustufen Bilder von Gesichtern. Jedes dieser Bilder wurde von je 10 Freiwilligen mithilfe von *Crowdsourcing* bewertet. Der Datensatz enthält für jede Klasse (Emotionen des *FACS* (inkl. neutral), "kein Gesicht" und "unbekannt") die Anzahl an Freiwilligen, welche das Bild entsprechend bewertet haben.
Ein Beispiel für ein einzelnes Datum des Datensatzes ist in Abbildung \ref{single_ferplus} zu sehen.

![Darsteillung einer einzelnen Zeile aus dem FER+ Datensatz. Oben ist das zugehörige Bild aus den Pixelwerten des originalen *FER2013* Datensatz zu sehen, darunter wird die zugehörige Zeile des *FER+* ausgegeben \label{single_ferplus}](source/figures/dump_ferplus.png){ width=90% } 

Das Team von Microsoft Research [@Barsoum2016] beschreibt mehrere Variationen wie die mehrfach *gelabelten* Daten verwendbar sind. In dieser Arbeit wird jedoch ausschließlich der einfache Mehrheits-Ansatz verfolgt. Es wird also jedes Bild der Klasse zugeordnet, welche die meisten Stimmen erhalten hat.

Zum Laden der Daten wurde das folgende Python Skript verwendet.

```python
def load_data_ferplus(fer_ds_path = "fer+/fer2013/fer2013.csv",
        ferplus_ds_path="fer+/fer2013new.csv"):
    # loading only the label cols from fer+
    cols=['neutral', 'happiness', 'surprise', 'sadness',
        'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    ferplus = pd.read_csv(ferplus_ds_path , usecols=cols,
        dtype=np.int32)
    # and only the pcitures (pixels) from original fer2013
    fer = pd.read_csv(fer_ds_path, usecols=['pixels'])
    # getting simple majority voted label using idxmax and
    # merge it with the picture dataset
    merged_fer = pd.concat([fer, ferplus.idxmax(axis=1)],
        axis=1)
    # redefining column names
    merged_fer.columns = ['img_pixels', 'emotion']
    return merged_fer
```

#### selbsterstellte Daten

Zur Generierung von eigenen Daten wurde im Rahmen der Arbeit eine einfache Website erstellt, welche mithilfe von *WebRTC* (Web Real-Time Communications) Zugriff auf die Kamera bekommt. Auf dieser Website haben freiwillige Probanden und auch der Autor dieser Arbeit die Möglichkeit, nacheinander für jede Emotion des *FACS* ein Video in der Länge von 15 Sekunden aufzunehmen. Dieses wird anschließend direkt auf dem Server gespeichert. Die Webseite ist in Abbildung \ref{webrtc_screenshot} zu sehen.

![Screenshot der Video-Recording Website\label{webrtc_screenshot}](source/figures/web_recorder.png){ width=70% }

Mithilfe dieser Webseite wurden insgesamt 20 Sätze von 8 verschiedenen freiwilligen Probanden (den Autor dieser Arbeit eingeschlossen) gesammelt. Aus diesen Videos wurde anschließend mit Hilfe des folgenden Python-Skripts pro Sekunde ein Einzelbild extrahiert und mit dem Namen der entsprechenden Klasse abgespeichert. Somit wurden also $20 * 15 = 300$ Einzelbilder pro Klasse generiert. 

```python
def extract_video_frames(prefix, videofile,
	targetdir = "../data/extracted"):

    vidcap = cv2.VideoCapture(videofile)
	fps = min((50, int(vidcap.get((cv2.CAP_PROP_FPS)))))
	print ("extracting every {} frame from {}".format(fps,
        videofile))
    success,image = vidcap.read()
    count = 0
    while success:
        if (count % fps) == 0:
			# save frame as JPEG file
            cv2.imwrite("{}/{}_frame{}.jpg".format(
                targetdir, prefix , count) , image)  
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
```
<!-- Python-Skript zum extrahieren der Einzelbilder\label{listing_extract_video} -->

Die Einzelbilder wurden anschließend manuell auf Korrektheit, das heißt Zuordnung zur Klasse, geprüft. Dabei wurden insgesamt 402 Bilder wieder aussortiert. 

## Datenpräparation \label{chapter_dataprep}

Um die Effizienz, sowie die Genauigkeit der Vorhersage des neuronalen Netzwerkes zu steigern, werden alle Daten bevor sie dem KNN präsentiert werden auf diverse Arten präpariert. Im Folgenden wird auf die angewandten Methoden genauer eingegangen.

### Vorverarbeitung

Um ein möglichst gutes Ergebnis zu erzielen, wurden die selbst erstellten Daten und die frei verfügbaren Daten aus dem *FER+* Datensatz auf diverse Weise vorverarbeitet. 
Bei den selbst aufgezeichneten Daten wurde, wie bereits erwähnt, eine abschließende manuelle Sichtung vorgenommen, um möglichst alle falsch markierten Daten auszusortieren. Des Weiteren werden alle selbst aufgezeichneten Bilder auf den Ausschnitt des Gesichtes beschränkt, bevor sie dem selbst entworfenen KNN präsentiert werden. Dazu wird der bereits beschriebene *Viola-Jones-Detektor* [@Shen1997] verwendet. Der das Gesicht enthaltende Teil des Bildes wird anschließend auf 48x48 Pixel skaliert und in ein Graustufenbild umgewandelt. Damit haben letzten Endes alle Eingangsdaten des neuronalen Netzes die gleiche Struktur. Der Python Code für diese Vorverarbeitung ist im folgenden Listing zu sehen.

```python
def prep_data_selfrecorded(selfrecorded_data, 
        viola_jones_model="facedetection/viola_jones.xml"):
    # creating facedetector
    face_detection = cv2.CascadeClassifier(viola_jones_model)
    img_input = selfrecorded_data['img'].tolist()
    imgs = []
    noface = []
    for i in range(len(img_input)):
        ## making it grayscale
        img = img_input[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #face detection 
        faces = face_detection.detectMultiScale(img,
            scaleFactor=1.1,minNeighbors=5,minSize=(48,48),
            flags=cv2.CASCADE_SCALE_IMAGE)
        # sorting out images with failed detection
        target_face = None
        if len(faces) == 0:
            print ("ERROR:",
                "there was no face detected on image,",
                "discarding it")
            noface.append(i)
            continue
        elif len(faces) > 1:
            print ("WARNING:",
                "there was more than one face",
                "detected on image,",
                "using the biggest one")
            for face in faces:
                size = face[2] * face[3]
                if target_face is None or (
                        target_face[2] * target_face[3]
                            < size):
                    target_face = face
        else:
            target_face = faces[0]
         ## cropping image to only face
        (fX, fY, fW, fH) = target_face
        img = img[ fY: fY + fH, fX: fX +fW].copy()
        #rescale it
        img = cv2.resize(img.astype('uint8'),(48, 48))
        imgs.append(img.astype('float32'))
    imgs = np.asarray(imgs)
    imgs = np.expand_dims(imgs, -1)
    ## dropping all failed detections from pd Frame
    selfrecorded_data = selfrecorded_data.drop(noface)
    ##returning emotions OneHot Encoded. 
    emo = pd.get_dummies(selfrecorded_data['emotion'])
    # we need to make sure, that we have always the same cols
    cols = ['anger', 'contempt', 'disgust', 'fear', 'happiness',
        'neutral', 'sadness', 'surprise' ]
    for col in cols:
        if col not in emo.columns.tolist():
            # if col not present create and fill with zeros
            emo[col] = 0
    return imgs, emo[cols].as_matrix()
```

Bei den Daten aus dem *FER+* Datensatz ist etwas weniger Vorverarbeitung nötig. Die Vorverarbeitung dieser Daten besteht im wesentlichen darin, die Pixelwerte, welche als ein eindimensionales Array vorliegen, in eine 48x48 Matrix umzuwandeln. Des Weiteren wird aus den mehrstimmigen *Labels* des *FER+* Datensatzes mithilfe des einfachen Mehrheitsprinzips das hier Genutzte extrahiert. In einem letzten Schritt werden alle Datensätze, welche mit "not a face" oder "unknown" markiert sind, aussortiert.

```python
def prep_data_ferplus(ferplus_data):
    # removing NF (no face) and unknown columns
    cleaned = ferplus_data[ferplus_data['emotion'] != "NF"]
    cleaned = cleaned[cleaned['emotion'] != "unknown"]
    ### Getting oneHot encoded classes using get_dummies
    emotions = pd.get_dummies(cleaned['emotion']).as_matrix()
    ## retrieving images from pixel list
    img_pixels = cleaned['img_pixels'].tolist()
    img_width, img_height = 48, 48
    imgs = []
    # extracting images from space delimited pixel values
    for pixel_row in img_pixels:
        img = [int(pixel) for pixel in pixel_row.split(' ')]
        # Having them in a one-dim list now,
        # we have to make an np array with our img_shape
        img = np.asarray(img).reshape(img_width, img_height)
        img = cv2.resize(img.astype('uint8'),(48, 48))
        imgs.append(img.astype('float32'))
    imgs = np.asarray(imgs)
    imgs = np.expand_dims(imgs, -1)
    return imgs, emotions
```


### Normalisierung

*"Data normalization has been proposed to address the aforementioned challenge by reducing the training space and making the
training more efficient."* [@Zhang2018]
<!--
Laut [@Zhang2018] hilft die Normalisierung also dabei die Herausforderung einer effizienten Erkennung zu meistern, indem das Trainingsspektrum verkleinert wird, was die Trainingsphase beschleunigt.-->

Ein üblicher Schritt um die Trainingsphase im maschinellen Lernen zu beschleunigen, ist es die Eingabedaten zu normalisieren. Ziel ist es die Eingabedaten, welche auf einem sehr breiten Spektrum liegen, zu normalisieren um das Spektrum zu verkleinern. Im vorliegenden Fall geht es um die Graustufenbilder. Im Generellen kann man die Normalisierung von solchen Bilden wie folgt beschreiben: Ein n-dimensionales Graustufenbild $I:\{\mathbb{X}\subseteq\mathbb{R}^n\}\rightarrow\{\text{Min},..,\text{Max}\}$ mit den Pixelwerten zwischen $Min$ unx $Max$ wird in ein neues Graustufenbild $I_N:\{\mathbb{X}\subseteq\mathbb{R}^n\}\rightarrow\{\text{newMin},..,\text{newMax}\}$ mit Pixelwerten zwischen $newMin$ und $newMax$ überführt.[@gonzalez2008digital]
Die lineare Normalisierung eines Graustufenbildes berechnet sich wie folgt:

$$
I_N=(I-\text{Min})\frac{\text{newMax}-\text{newMin}}{\text{Max}-\text{Min}}+\text{newMin}
$$

Im vorliegenden Beispiel sind die Ausgangswerte für $Min = 0$ und $Max = 255$ und für eine einfache Normalisierung werden die Werte $newMin = 0$ und $newMax = 1$ gewählt, damit alle Datenwerte zwischen 0 und 1 liegen. Damit ergibt sich folgende vereinfachte Formel:

$$
I_N=\frac{I}{\text{Max}} \Rightarrow I_N=\frac{I}{255}
$$

Zur Normalisierung der Daten werden dementsprechend alle Pixelwerte durch 255 dividiert, bevor das Bild dem neuronalen Netz gezeigt wird.

### Datenmehrung

Je mehr Trainingsdaten für das KNN vorhanden sind, desto besser kann es auch mit ungesehenen Daten umgehen. Da nur begrenzt viele Daten zur Verfügung stehen werden, in dieser Arbeit einige Methoden der künstlichen  Datenvermehrung angewandt. Dazu werden die Bilder der Eingangsdaten zum Beispiel gespiegelt, verzerrt oder gedreht. In dieser Arbeit wurden die Methoden der Spiegelung, sowie des zufälligen Drehens einiger Bilder angewandt. Durch die Spiegelung, bzw. Drehung eines Bildes entsteht wieder ein neues Bild, welches zum Training des neuronalen Netzes verwendet werden kann. So kann mit dieser relativ einfachen Methode die Anzahl der Trainingsdaten sehr leicht verdoppelt werden. Zur Datenmehrung während des Trainingsprozesses wurde der *ImageDataGenerator* aus dem *keras* Modul[@Keras.io2019] verwendet. Dieser bietet die Möglichkeit zufällige Bilder zu spiegeln oder zu rotieren. Dazu wurden die Parameter *rotation_range* und *horizontal_flip* entsprechend gesetzt.

```python
img_gen = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=45,
                        horizontal_flip=True)
```

### Einteilung der Datensätze \label{chapter_dataprep}

Beim maschinellen Lernen ist es üblich den vorhandenen Datensatz bzw. die vorhandenen Datensätze in verschiedene Verwendungszwecke einzuteilen. Klassisch spricht man hier vom *train/test-Split*, also einer Aufteilung der Daten in einen Trainings- und einen Test-Datensatz. In modernen Projekten, welche sich mit maschinellen Lernen beschäftigen, spricht man jedoch zumeist von einem *train/dev/test-split*. Die Daten werden also in einen Trainings-, einen Entwicklungs- und einen Test-Datensatz eingeteilt. Als Entwicklungs-Datensatz bezeichnet man jene Daten, welche während der Entwicklung, also dem Anpassen bestimmter (Hyper-)Parameter, des neuronalen Netzes zur Evaluierung verwendet werden. Der Test-Datensatz ist in diesem Szenario ein Satz aus Daten, welches das neuronale Netz vor der Fertigstellung noch nicht "zu sehen" bekommen hat. Beim klassischen *train/test-split* ist der Test-Satz also eigentlich das, was wir heute als Entwicklungs-Datensatz bezeichnen und es gibt keinen wirklichen Test-Datensatz.
Bei der Wahl der Datenquellen ist es wichtig, dass die Test-Daten möglichst ähnlich zu den später erwarteten Eingangsdaten sind und die Entwicklungs- und Test-Datensatz aus der selben Quelle stammen.

Für diese Arbeit bedeutet das, dass die Entwicklungs- und Test-Daten aus den selbstgenerierten Daten stammen, da diese bereits von aufgenommenen Videos stammen, was den Zeildaten sehr nahe kommt.
Als Trainingsdaten wird entsprechend der *FER+* Datensatz verwendet.
Ein Problem bei einer solchen Aufteilung, wenn also die Trainingsdaten aus einem anderen Datensatz stammen als die Entwicklungs und Test-Daten, ist, dass man gewisse Probleme, wie zum Beispiel eine Überanpassung, teilweise nur schwer erkennen kann. Deshalb ist es in einem solchen Fall sinnvoll noch einen vierten Datensatz einzuführen, welcher aus der selben Quelle wie die Trainingsdaten stammt (hier *FER+*). Man spricht hier vom *dev_train* oder auch *bridge* Datensatz. Dieser wird im Prinzip analog zum Entwicklungsdatensatz behandelt und dient zum Testen der Parameter des neuronalen Netzwerkes nach jeder Änderung. Anhand der Unterschiedlichen Ergebnisse für den *bridge* und den *dev* Datensatz kann man nun schnell bestimmte Probleme des neuronalen Netzwerks erkennen.

In dieser Arbeit wurde daher auch die Einteilung in 4 Datensätze gewählt. Die Daten wurden dabei wie folgt aufgeteilt:
Die selbsterstellten Daten wurden in zu je 50% in den Entwicklungs- und Test-Datensatz aufgeteilt. Vom *FER+* Datensatz wurden 10% der Bilder für den *Bridge* Datensatz verwendet und 90% als Trainingsdaten. Die Aufteilung ist in Abbildung \ref{data_split} veranschaulicht.

![Aufteilung der Daten in 4 Datensätze \label{data_split}](source/figures/train_test_split.pdf){ width=90% } 

Zum Aufteilen der einzelnen Datensätze wurde die Funktion "train_test_split" aus dem Python Modul "sklearn"[@scikit-learn] verwendet. Um eine zwar anfangs zufällige, jedoch reproduzierbare Aufteilung zu erhalten, wird der "Random_state" auf einen festen Wert gesetzt. Das genutzte Python Skript ist im Folgenden abgebildet.

```python
def split_datasets(ferplus_imgs, ferplus_emotions,
        selfrecorded_imgs , selfrecorded_emotions ):
    xTrain, xBridge, yTrain, yBridge = train_test_split(
        ferplus_imgs, ferplus_emotions, test_size = 0.1,
            random_state = 20808)
    xDev, xTest, yDev, yTest = train_test_split(
        selfrecorded_imgs, selfrecorded_emotions,
            test_size = 0.5, random_state = 280919)
    datasets = [xTrain, xBridge, yTrain, yBridge,
        xDev, xTest, yDev, yTest]
    return datasets
```

## Entwurf und Entwicklung neuronaler Netze

Im Rahmen dieser Arbeit sollen 2 verschiedene neuronale Netzwerke entworfen, trainiert und evaluiert werden. Alle beide sollen gefaltete neuronale Netze sein, sich jedoch in der Topologie und den (Hyper-) Parametern unterscheiden.

### Entwicklungsumgebung

Die Entwicklung der KNN´s wurde in Python mithilfe der Machine-Learning Erweiterunge Keras[@Keras.io2019] vorgenommen. Keras ist eine vereinfachte Schnittstellenimplementierung zur einfachen Verwendung von verschiedenen Machine Learning-Schnittstellen. In dieser Arbeit wurde Keras mit der Tensorflow-Schnittstelle verwendet.
Als Entwicklungsumgebung wurde hierzu ein Jupyter Notebook verwendet. Der Vorteil eines Jupyter Notebook liegt darin, dass sehr einfach Text und Programmierabschnitte, sowie deren Ausgabe nebeneinander visualisiert werden können.

### Topologie

Unter der Topologie des KNN versteht man die Architektur im Zusammenhang mit den (Hyper-)Parametern. Die Architektur beschreibt den Aufbau oder die Struktur des Netzwerkes. In einem CNN also im wesentlichen die Art und Reihenfolge der einzelnen Netzwerkschichten.

Als Hyperparameter hingegen bezeichnet man weitere Rahmenparameter, welche unabhängig vom grundlegenden Aufbau des Netzes verändert werden können. Einige solcher Hyperparameter wurden in Kapitel 2 bereits vorgestellt. Oft werden auch die Größe oder die Anzahl der wiederholten Schichten als weitere Hyperparameter betrachtet. Der Grundgedanke beim entwerfen eines neuronalen Netzwerkes ist das finden der besten ameter für ein optimales Ergebnis.

In dieser Arbeit sollen zwei verschiedene Netzwerktopologien für das Problem der Emotions-Klassifizierung entworfen und verglichen werden.

#### Einfaches faltendes neuronales Netz

Als erstes Modell soll ein sehr einfaches faltendes neuronales Netz entworfen werden.

##### Architektur

Das Neuronale Netzwerk besteht aus insgesamt 4 *Faltungs-Stapeln*, gefolgt von einer Ausgabeschicht. Die *Faltungs-Stapel* bestehen jeweils aus zwei Faltungsschichten, auf welchen, abgesehen von der letzten Schicht, eine Stapel-Normalisierung (engl. batch normalization) angewandt wird, gefolgt von einer Pooling Schicht, auf welche eine *Dropout*-Regularisierung angewandt wird. In Abbildung \ref{architecture_simple_cnn} ist die Architektur kurz dargestellt.

![Architektur des einfachen Faltungsnetzwerekes. \label{architecture_simple_cnn}](sources/figures/simple_cnn_arch.pdf){width: 80%}

##### Hyper-Parameter

Die Topologie des KNN lässt sich durch die folgend beschriebenen Parameter konkretisieren.

* **Filteranzahl**: Die Anzahl der Filter in den einzelnen Faltungsschichten beeinflusst die Dimension der Folgedaten. Für dieses Netzwerk wurde die Filteranzahl für beide Faltungsschichten eines Stapels in der Regel gleich gesetzt. Mit jedem Stapel verdoppelt sich die Anzahl der Filter. Lediglich die letzte Faltungsschicht weicht von diesem Schema ab und verwendet die fixe Anzahl von 8 Filtern, was der Anzahl der Klassen $|Z|$ entspricht. Die Anzahl der Filter der ersten Faltungsschicht wurde auf 16 festgelegt. Die Schichten in den darauffolgenden Stapeln verwenden also jeweils 32, 64, und 128 Filter.

* **Filtergröße**: Auch die Filtergröße hat einen nicht zu vernachlässigenden Einfluss auf die Dimension der Daten. In diesem Netzwerk wurde für den ersten Stapel eine Filtergröße von $7x7$ festgelegt. Für die 2 folgenden Stapel wurde die Filtergröße jeweils um $2x2$ verringert und beträgt damit $5x5$, bzw. $3x3$. Der letzte Stapel verwendet ebenfalls eine Filtergöße von $3x3$.

* **Füllung**: Die Füllung (engl. padding) bei einer *convolutional* Schicht beeinflusst im Zusammenspiel mit der Filtergröße die Dimension der Ausgabedaten. In diesem Modell wurde für jede Faltungsschicht das sogenannte *same padding* verwendet. Das bedeutet, dass genau so viele Zeilen aufgefüllt werden, die nötig sind, damit die zweidmensionale Ausgabedimension der Eingabedimension entspricht. Da sich im Allgemeinen die Dimension der Ausgabe einer neuronalen Faltungsschicht mit einer Eingabedimension von $n x n$ mit einem *padding* von $p$ und einer Filtergröße von $f x f$ wie folgt berechnen lässt. 

$$
conv(n x n, f x f, p) = n +2p - f + 1 x n +2p - f + 1
$$

$$
\text{Ergibt sich für das "same padding" die Größe der Füllung } p \text{ wie folgt.}
$$

$$
n + 2p_{same} -f +1 = n
$$
$$
p_{same} = \frac{f-1}{2}
$$

* **Pooling**: Als Pooling Methode wird in diesem Netzwerk das Durchschnitts-Pooling (engl. *average pooling*) angewandt. Beim Average Pooling, wird ähnlich wie bei einer *convolutional* Schicht, ein Filter einer definierten Größe (hier $2x2$) über die Daten *geschoben*. Beim *average pooling* erhält Das Ziel-Fenster immer den Durchschnittswert, der Werte im Fenster (Vergleich Abbildung \ref{avg_pooling}). Die letzte *pooling* Schicht in diesem Netz stellt eine Besonderheit dar, da hier da sogenannte globale Durchschnitts-Pooling verwendet wurde. Beim *global average pooling* wird für jede zweidmensionale Ebene genau ein Durchschnittswert gebildet, die Filtergröße ist also gleich der Größe der Eingangsdaten.

![Beispielhafte Darstellung von *Average-Pooling* mit einer Fenstergröße von 2x2 und einer Schrittgröße(stride) von 2 \label{avg_pooling}](sources/figures/avg_pooling.pdf){width: 80%}

* **Kosten-Funktion**: Zur Ermittlung des Verlustes wurde die kategorische Kreuzentropie-Funktion verwendet. Für die genaue Funktionsweise sei auf [@Litomisky2012] verwiesen, da sie in dieser Arbeit nicht genauer behandelt wird.

<!-- KERSTINs-Korrektur bis her -->

#### *Transfer Learning* auf Basis XCeption

Beim zweiten Model wurde auf die sogenannte *transfer learning* Technik gesetzt. Darunter versteht man das übertragen der Gewichte eines bereits trainierten Netzwerkes auf eine neue, meist ähnliche, Aufgabe. Dabei wird in der Regel die Ausgabeschicht des ursprünglichen Netzes entfernt und eine oder mehrere neue vollvernetzte Schichten angehängt, welche dann die Aufgabe haben vom ursprünglichen auf das neue Problem zu schließen.
In dieser Arbeit wurde *transfer learning* auf Basis des XCeption Netzwerkes[@Chollet2017] welches für die Klassifizierung des *ImageNet* Datensatzes[@ILSVRC15] trainiert wurde. Für das XCeption Netzwerk wurde sich aufgrund der Tatsache entschieden, dass es für den *ImageNet* Datensatz eine sehr hohe Genauigkeit mit einer relativ geringen Anzahl an Parametern (siehe Tabelle 3.1) erzielen konnte.

 Model | Top-1 Genauigkeit | Top-5 Genauigkeit | Parameter-Anzahl |
| ----- | --------------: | --------------: | ----------: |
| Xception| 0.790 | 0.945 | 22,910,480 |
| VGG16 | 0.713 | 0.901 | 138,357,544 |
| VGG19 | 0.713 | 0.900 | 143,667,240 |
| ResNet50 | 0.749 | 0.921 | 25,636,712 |
| ResNet101 | 0.764 | 0.928 | 44,707,176 |
| ResNet152 | 0.766 | 0.931 | 60,419,944 |
| ResNet50V2 | 0.760 | 0.930 | 25,613,800 |
| ResNet101V2 | 0.772 | 0.938 | 44,675,560 |
| ResNet152V2 | 0.780 | 0.942 | 60,380,648 |
| ResNeXt50 | 0.777 | 0.938 | 25,097,128 |
| ResNeXt101 | 0.787 | 0.943 | 44,315,560 |
| InceptionV3 | 0.779 | 0.937 | 23,851,784 |
| InceptionResNetV2  | 0.803 | 0.953 | 55,873,736 |
| MobileNet | 0.704 | 0.895 | 4,253,864 |
| MobileNetV2 | 0.713 | 0.901 | 3,538,984 |
| DenseNet121 | 0.750 | 0.923 | 8,062,504 |
| DenseNet169 | 0.762 | 0.932 | 14,307,880 |
| DenseNet201 | 0.773 | 0.936 | 20,242,984 |
| NASNetMobile | 0.744 | 0.919 | 5,326,716 |
| NASNetLarge  | 0.825 | 0.960 | 88,949,818 |

Tabelle 3.1: Vergleich verschiedener vortrainierter Modelle des *Keras* Frameworks. Die Top-1 und Top-5 Genauigkeit wurden für den *ImageNet* Datensatz erzielt. - Quelle: [@Keras.io2019]  \label{table_31} 



##### Architektur

Das Xception Netzwerk basiert auf sogenannten *separable convolution* Bausteinen. Diese bestehen aus einer Schicht an räumlichen Faltungen (zum Beispiel 3x3 *convolution*) pro Eingangskanal, gefolgt von einer punktweisen Faltung (1x1 *convolution*).

![vereinfachte Darstellung einer *separable convolution*. Zuerst wird eine *depthwise convolution*  (3x3 Filter) durchgeführt. Im Anschluss die *pointwise convolution* (1x1 Filter) \label{sepconv}](source/figures/xception_architecture.png){ width=80% }

Es besteht aus insgesamt 36 Faltungsschichten, welche in 14 Module eingeteilt wurden. Eine Darstellung der usrpünglichen Architelktur ist in Abbildung \ref{xception_architektur} zu sehen.

![Architektur des ursprünglichen Xception Netzwerkes. Die Eingabedaten durchlaufen zunächst den *Entry flow* und werden dann an den *Middle flow* weitergegeben. Dieser wird insgesamt 8 mal wiederholt bevor die Daten an den *Exit flow* weitergereicht werden. Quelle: [@Chollet2017] \label{xception_architektur}](source/figures/xception_architecture.png){ width=80% }

Für das *Transfer Learning* wurde die Ausgabeschicht des ursprünglichen XCeption Netzwerkes entfernt. Danach wurde immer eine vollvernetzte Schicht, auf welche die Dropout Regularisierung angewendet wurde sowie eine variable Anzahl an weiteren vollvernetzten Schichten (inkl. Dropout) angehängt. Zur Ausgabe wurde eine Softmax Schicht verwendet. Die Anzahl und Größe der vollvernetzten Schichten wurden als Hyperparameter während des Trainings betrachtet.

##### Datenanpassung

Die kleinste Dimension der Eingabdaten welche das XCeption Netzwerk akzeptiert ist $71 x 71 x 3$. Daher musste die Datenvorbereitung für dieses Netzwerk angepasst werden. Da die Daten aus dem FER+ Datensatz nur als Graustufenbild im Format 48 x 48 vorliegen wurden diese mithilfe der Python Bibiliothek *open-cv* in 71 x71 BGR Bilder konvertiert um die Zieldimension zu erhalten. 
Bei den selbsterstellten Daten wurde lediglich auf die Konvertierung in ein Graustufenbild verzichtet und die Größe auf 71 x 71, anstatt 48 x 48 angepasst.

## Training und Evaluierung

Nach dem Festlegen der Netzwerktopoligien begann die Trainingsphase. Hierzu wurden die vorher festgelegten Trainings, *Bridge* und Entwicklungs-Datensätze herangezogen. Beim Trainieren der Netzwerke gibt es diverse weitere (Hyper-)Parameter, welche sich auf die Leistung des Netzes auswirken. Einige dieser Parameter wurden in dieser Arbeit konstant gehalten, andere jedoch wurden variabel gehalten und so für die Optimierung des Klassifizierers genutzt. Für jedes Netzwerk wurden mehrere Trainingsdurchläufe gestartet und die Leistung des Netzes überprüft um die bestmögliche Belegung dieser Parameter zu finden.

### konstante Trainingsparameter

* **Stapelgröße (engl. *batch size*)**: Die Stapelgröße legt fest, wie viele Trainingsdaten dem Netzwerk auf einmal präsentiert werden. Je größer die Stapelgröße gewählt wird, desto schneller kann das Netzwerk trainiert werden, da die Eingangsdaten in einer Matrix zusammen gefasst werden. Jedoch wird pro Stapel auch immer nur eine Anpassung der Gewichte vorgenommen. Die Stapelgröße wurde in dieser Arbeit für alle Trainingsdurchläufe konstant auf 128 gesetzt.

* **Lernalgorithmus**: Als Lernalgorithmus wurde der *ADAM* Algorithmus gewählt [@Kingma2014]. Die Parameter $\beta_1$, $\beta2$ und $\epsilon$ des *ADAM* Algorithmus wurden bei allen Traininsdurchläufen mit den Standardwerten, also $\beta_1=0,9$, $\beta_2=0,999$ und $\epsilon=10^{-8}$, initialisiert. Die Lernrate $\alpha$ wurde als separater Parameter behandelt.


* **reduzieren der Lernrate**: Das dynamische reduzieren der Lernrate wird verwendet, wenn sich eine bestimmte Metrik (hier der Netzwerkfehler für den Entwicklungs-Datensatz) eine bestimmte Anzahl an Epochen nicht verbessert hat. Tritt dieser Fall ein wird die Lernrate um einen bestimmten Faktor angepasst. Die Anzahl der Epochen ohne Verbesserung wird durch den *Patience*-Parameter $p$ beschrieben. Dieser wurde in dieser Arbeit immer fix mit $p = 20$ intialisiert. Die neue Lernrate errechnet sich dann anhand des Anpassungsfaktors $\lambda$ wie folgt:

$$
    \alpha_{neu} = \alpha * \lambda
$$

$$
\text{Die Anpassungsrate wurde in dieser Arbeit fix auf }\lambda = 0,1 \text{ gesetzt.}
$$

* **vorzeitiges Trainingsende (engl. *early stopping*)**: Unter *early stopping* versteht man das frühzeitige Abbrechen eines Trainingsvorganges um eine Überanpassung und damit verbundene schlechtere Generalisierung zu verhindern. Bei den Trainingsdurchläufen im Rahmen dieser Arbeit wurde nicht direkt *early stopping* verwendet, jedoch wurde folgendes Verfahren angewendet. Nach jeder Epoche des Trainings wurde der *Netzwerkfehler* für die Daten aus Entwicklungs-Datensatz berechnet. Sobald sich dieser im Vergleich zur vorherigen Epoche verbessert hat wurde der Zustand des Netzwerkes (Also die Gewichte) abgespeichert. Somit konnte am Ende des Trainings der Trainingstand, mit dem besten Ergebnis für den Entwicklungs-Datensatz, gewählt werden, unabhängig von der gewählten Anzahl an Trainingsepochen.


* **Lernrate**: Für die Lernrate $\alpha$ wurden im Vorfeld mehrere verschiedene Werte evaluiert. Es wurden insgesamt 4 verschiedene Werte auf dem logarithmischen Interval zwischen $10^{-4}$ und $10^{-1}$ getestet. Die besten Ergebnisse erzielte der Wert $\alpha = 10^{-3} = 0,001$ weshalb dieser in allen zukünftigen Trainingsläufen verwendet wurde.

### variable Trainingsparameter

* **Anzahl der Epochen**: Die Anzahl der Epochen, legt fest wie oft der gesamt Trainingsdatensatz dem Netzwerk präsentiert wird und die Gewichte entsprechend angepasst werden. Zu Anfang wurden die Netze jeweils für 10 Epochen traininert. Anhand der Trainingsverläufe wurde zunächst versucht, alle  weiteren Parameter zu optimieren. Nachdem ein viel versprechendes Set an Parametern gefunden wurde, wurde das Netz mit diesen für 20 Epochen trainiert und eine erneute Anpassung vorgenommen. So wurde die Anzahl der Trainings-Epochen immer nach oben iteriert, bis zu 150 Epochen.

Die folgenden Paramter wurden nur beim zweiten Model verwendet.

* **Anzahl der neu zu trainierenden Schichten:** Üblicherweise werden beim *transfer learning* alle Schichten des ursprünglichen Netzwerkes gesperrt. Das heißt, die Gewichte dieser Schichten sind während des Trainings statisch und werden nicht angepasst. Oft bringt es jeodch ein gutes Ergebnis, wenn man ein paar der hinteren Schichten für das Training öffnet. Der Hintergrund ist, dass die ersten Schichten nur Basis-Merkmale erkennen. Je tiefer die Schicht jedoch ist, desto detailierter ist das Merkmal, dass Sie extrahieren. Da man beim *transfer learning* das gelernte eines Problems auf ein anderes anwendet, kann man mit dem öffnen der hinteren Schichten seine Ergebnisse oft verbessern. Die Anzahl der letzten Schichten des Xception Netzes, welche neu traininert wurden wurde als Parameter behandelt, welcher in der Trainingsphase die Werte 3, 6, 9, 12, 15 und 18 angenommen hat. 

* **Größe der ersten vollvernetzten Schicht**: Die Größe der vollvernetzten Schicht, welche am Ende des Xception Netzwerkes angehängt wurde wurde als weiterer Parameter betrachtet. Hier wurden die Werte 128 und 64 verwendet.

* **Anzahl der zusätzlichen vollvernetzten Schichten**: Es wurden variabel zusätzliche vollvernetzte Schichten zwischen der ersten vollvernetzten Schicht und der Ausgabeschicht eingebaut. Die Größe dieser Schichten wurde konstant mit 32 gewählt. Nach jeder der vollvernetzten Schichten, abgesehen von der letzten, wurde die Dropout Regularisierung angewendet. Für diesen Parameter wurden Werte zwischen 0 (keine zusätzliche Schicht) und 5 gewählt.

## Optimierung der Parameter

Zunächst wurden die Netzwerke jeweils für eine geringere Anzahl an Epochen traininert. Hierbei wurde versucht eine möglichst gute Genauigkeit für die eigentlichen Trainingsdaten zu erlangen. Nach Optimierung wurden auch längere Trainingsläufe durchgeführt.

### einfaches Faltungsnetzwerkes

Für das einfache Faltungsnetzwerk wurden hier nur wenig befriedigenden Ergebnisse erzielt (siehe Abbildung \ref{simple_cnn_training}). Die Genauigkeit der Trainingsdaten stagnierte hier schnell bei einem Wert um $0,65$. Daher wurde sich im weiteren Verlauf auf die Optimierung des zweiten Netzes (*transfer Learning* auf Basis des Xception Netzes) beschränkt.

![Genauigkeit während des Trainingsverlaufs des einfachen Faltungs-Netzwerkes für 50 Epochen. Man erkennt ein *Bias* Problem, da das Netzwerk, selbst den Trainingsdatensatz nicht ausreichend gut erlernen kann. \label{simple_cnn_training}](source/figures/simple_cnn_training.png){ width=80% }

### *Transfer Learning* Netzwerk

Für das *Transfer Learning Netzwer* wurden diverse Trainigndurchläufe mit den verschienden Paramtern durchgeführt. Zusammenfassend kann man sagen, dass alle der trainierten Netzwerke, ab einem bestimmten Punkt eine Überanpassung an den Trainingsdatensatz erreicht haben.

Der Trainingsverlauf des besten gefundenen Netzes, das heißt mit der besten Test- und Entwicklungs-Genauigkeit ist in Abbildung \ref{best_xcepton_training} dargestellt. Es handelt sich dabei um das Model mit 64 Einheiten in der ersten vollvernetzten Schicht und einer zusätzlichen vollvernetzten Schicht. Es wurden die letzten 12 Schichten des ursprünglichen Xception Netzwerkes neu traininert und die Anzahl der Epochen betrug 100. Das beste Modell wurde von Epoche 30 (*early stopping*) gewählt.

![Genauigkeit während des Trainingsverlaufs des besten gefunden Models für 100 Epochen. Man sieht, dass bereits relativ früh eine Überanpassung an den Trainingsdatensatz stattfindet. \label{best_xcepton_training}](source/figures/best_xception_training.png){ width=80% }

Wie man sieht ist auch das beste gefundene Model leider keine optimale Lösung für das Problem. An der Abweichung zwischen dem *Bridge* und dem Trainingsdatensatz lässt sich gut ein Generalisierungsproblem erkennen, also eine Überanpassung des Netzes an die Trainingsdaten. In der Regel gibt es drei verschiedene Ansätze um diesem entgegen zu wirken.

1. Trainieren eines kleineren Netzwerkes.
2. Trainieren des Netzes mit mehr Trainingsdaten.
3. Generalisierung der Daten.

Es kann in der Tat helfen ein kleineres Netzwerk zu trainieren um einer Überanpassung entgegenzuwirken. Dies hat jedoch den Nachteil, dass sich mit einer Verkleinerung auch die Genauigkeit verschlechtert. Da auch diese im obigen Modell mit 70% nicht perfekt ist, wurde dieser Ansatz nicht gewählt.

Auch der zweite Ansatz konnte leider nicht gewählt werden, da keine Möglichkeit bestand im zeitlichen Rahmen der Arbeit einen größeren Trainings-Datensatz zu erlangen.

Eine Regularisierung der Daten wurde von Anfang an durchgeführt, indem zum Beispiel eine Varieierung der Eingangsbilder mithilfe des *Keras ImageGenerator* vorgenommen wurde. Um die Daten im Netzwerk noch weiter zu Regularisieren wurden weitere kleine vollvernetzte Schichten an das Netz gehängt, auf welche eine Dropout Regularisierung angewendet wurde.
Diese Methode konnte kleinere Erfolge erzielen um der Überanpassung entgegenzuwirken. Jedoch hatte auch diese Methode einen negativen Einfluss auf die Genauigkeit des Netzes auf den Trainingsdatensatz (Vergleich Abbildung \ref{more_regularization_acc} und \ref{more_regularization_loss}).

![Genauigkeit whärend des Trainingsverlaufs eines Models mit zusätzlichen vollvernetzten Schichten inkl. Dropout Regularisierung \label{more_regularization_acc}](source/figures/more_reg_acc.png){ width=80% }

![Verlust während des Trainingsverlaufs eines Models mit zusätzlichen vollvernetzten Schichten inkl. Dropout Regularisierung \label{more_regularization_loss}](source/figures/more_reg_loss.png){ width=80% }

Eine weitere Tatsache, welche aus den Auswertungen der Trainingsverläufe hervorgingen ist ein sogenanntes *data mismatch* Problem. Das bedeutet, dass die Daten aus dem Entwicklungs-Datensatz, den Daten aus dem Trainingsdatensatz offenbar nicht ähnlich genug sind. Dieses Problem liegt zum einen an der Verschiedenheit der selbsterstellten Daten an sich, da sie zum Beispiel mit unteschiedlichen Kameras aufgenommen wurden, etc., als auch an der Qualität des FER+ Datensatzes. Dieser enthält zum einen nur relativ niedrig auflösende Graustufenbilder und zum anderen auch teilweise für das Netz verwirrende Daten, wie zum Beispiel Gesichter von Comic-Figuren anstatt von lebenden Personen.



## Entwicklung des eines Webservice

Auch wenn das beste Model, noch nicht die perfekte gewünschte Genauigkeit erzielen konnte, wurde es verwendet um einen Prototyp eines Webservice zu entwicklen.
Nachdem sich für das passende neuronale Netz entschieden wurde wird dieses einem Webservice zur Verfügung gestellt. Der Webservice hat die Aufgabe Videodaten entgegen zu nehmen und sekundenweise Einzelbilder an den Klassifizierer zu übergeben und anhand der Ausgabe eine Zeitleiste mit den erkannten Emotionen im JSON (JavaScript Object Notation) Format zurück zu liefern.

Der Webservice wurde mithilfe der Python Erweiterungen Flask und connexion realisiert.
Flask ist eine schlanke Erweiterung zur einfachen Erstellung von Web Diensten in Python. Connexion ist eine Erweiterung welche auf Flask aufbaut um API (Application Programmable Interface) Endpunkte anhand von einer OpenAPI[@OpenAPI] Spezifikation zu generieren.


Der Werbservice, sowie der Klassifizierer sind als sogenannte Microservices aufgebaut welche in separaten *Containern* laufen. (siehe Abbildung \ref{app_architecture}). Der Klassifizierer wird dem Webservice mithilfe von Tensorflow-Serve zur Verfügung gestellt. 

![Übersicht über die Software Architektur des entwickelten Webservice. Der Web Endpunkt und der Klassifizierer sind als Microservices konzipiert und kommunizieren über HTTP miteinander. \label{app_architecture}](source/figures/app_architecture.pdf){ width=80% }

Damit dies funktioniert musste das fertige Model zunächst vom Keras Datenformat in das Tensorflow-eigene Datenformat konvertiert werden. Dies wurde mit dem folgenden Python Skript erledigt.

```python
import tensorflow as tf
import os

def keras2tf(model_path = "models/keras/model.hdf5",
        export_path="models/tf/", model_version = 1):
    #extract model name from path
    model_name = os.path.splitext(
        os.path.basename(model_path))[0]
    # create output path
    export_dir = export_path + model_name
    os.mkdir(export_dir)
    # create version for tf serve
    export_dir = export_dir + str(model_version)
    ## deactivate learning phase
    tf.keras.backend.set_learning_phase(0)
    ##load keras model
    model = tf.keras.models.load_model(model_path)
    # exporting as tf model
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_dir,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})
```

Der Webservice nimmt die Videodaten im Base64[@Base64] Format entgegegen und extrahiert jede Sekunde ein Einzelbild. Dieses sendet er an den Klassifizierer um die enstprechende Emotion hervorzusagen und speichert diese zusammen mit der Sekunde ab um Sie zurückzugeben. Für die Verarbeitung des Videos wird die gleiche Python Funktion verwendet, die auch schon zur Vorverarbeitung der selbsterstellten Daten verwendet wurde. Die Einzelbilder werden vom Webservice dann mit folgender Funktion einer Emotion zugeordnet.
```python
def pred_to_text(pred, cols = ['anger', 'contempt',
         'disgust', 'fear', 'happiness', 'neutral', 
         'sadness', 'surprise' ]):
    
    # getting the emotion label for the highest predicition
    highest = 0.0
    emotion = ""
    for i in range(len(pred['predictions'][0])):
        if(highest < pred['predictions'][0][i]):
            highest = pred['predictions'][0][i]
            emotion = cols[i]
    return emotion

def predict_emotion(image):
    img = prep_single_img_selfrecorded(image,
        image_shape=(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT),
            grey=INPUT_IMG_GREY)
    
    if img is None:
        return "no_face"
    img = normalize_input(img)
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }
    # sending post request to TensorFlow Serving server
    r = requests.post(TF_CLASSIFIER_URL, json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    return pred_to_text(pred)
```

<Alle Micro Services wurden mithilfe von *Docker* und *docker-compose* containerisiert. Auf die genauere Beschreibun von Docker soll in dieser Arbeit nicht weiter eingegangen werden (Mehr Informationen unter [@Docker]). Docker-compose stellt eine einfache Anwendung zum festhalten aller Parameter einer aus *Docker*-Container bestehenden Microservice-Architektur dar. Dabei werden alle Komponenten in eriner Datei im YAML (Yet another Markup Language) Format festgehalten. Beim YAML Format handelt es sich vereinfacht gesagt um eine für den Menschen leichter lesbare Abwandlung des JSON Formates.<!-- Die in dieser Arbeit erstellte Architektur wurde in einer *docker-compose* Datei festgehalten. -->
<!-- move to anhan 
```yaml
version: '3.0'
services:
  api:
    image: skraus/emopred-api
    build: ./api
    environment:
      TF_CLASSIFIER_URL: http://classifier:8501/v1/models/model:predict
    ports:
      - 8080:8080
  classifier:
    image: skraus/emopred-classifier
    build: ./classifier
    environment:
      MODEL_NAME: model
```
-->

Zum Testen der Funktionalität wurde außerdem eine sehr minimalistische Benutzerschnittstelle mithilfe von HTML (Hyper Text Markup Language) und Javascript erstellt, welche die Funktionalität bietet eine Video-Datei an den Webservice zu senden und die Ergebnisse im JSON Format anzuzeigen (siehe Abbildung \ref{frontend})

![Screenshot der minimalistischen Benutzerschnittstelle \label{frontend}](source/figures/screenshot_frontend.png){ width=80% }

