\clearpage

# Umsetzung & Evaluierung

In diesem Teil wird die Umsetzung der Arbeit beschrieben. Im ersten Abschnitt wird auf die Art und Einteilung der zu klassifizierenden Daten eingegangen. Im weiteren Verlauf wird die Beschaffung der verschiedenen Datensätze behandelt, mit welchen gearbeitet werden soll. Anschließend werden diverse Verfahren der Datenpräparation wie Vorbereitung, Normalisierung und Datenvervielfältigung thematisiert. Letzten Endes werden verschiedene neuronale Netze entworfen, trainiert und getestet, bis dann im letzten Abschnitt die Realisierung des zu erstellenden Webservice näher betrachtet wird.

## Beschaffenheit und Einteilung der Daten

Wie bereits beschrieben, sollen für den späteren Webservice Videodaten als Eingabe dienen. Das neuronale Netz wird jedoch nicht direkt die Videodaten, sondern aus dem Video extrahierte Einzelbilder als Eingabe erhalten. Diese Eingabedaten werden anhand später beschriebener Verfahren noch weiter vorbereitet und optimiert.
Die Einzelbilder sollen als *Array* von Pixelwerten eines Graustufenbildes an das neuronale Netzwerk übergeben werden, siehe Kapitel Datenpräparation <!-- TODO: REF -->.
Die Bilder sollen in 8 verschiedene Klassen eingeteilt werden. Die verschiedenen Emotionen des *FACS* und die zusätzliche Klasse *neutral* bilden den Zielklassenvektor $Z$, es gilt also $|Z| = 8$. Jedes Bild, welches der selben Emotion des *FACS* entspricht, ist Mitglied der selben Klasse aus $Z$.

## Datensätze

Ein Teil der Arbeit bestand darin, geeignete Datensätze für das Training und die Evaluierung des neuronalen Netzes zu finden und diese später in entsprechende Trainings, Evaluierungs und Test-Datensätze für das neuronale Netzwerk zu unterteilen.

### Beschaffung der Datensätze

Dazu sollen 2 verschiedene Ansätze unterschieden werden. Zum einen die Beschaffung eines vorhandenen freien Datensatzes von Gesichtsbildern inklusive der Zuordnung zu einer der entsprechenden *FACS* Emotionen, sowie zum anderen die Generierung von eigenen Daten mithilfe eines Webservice und freiwillgen Probanden. 
In dieser Arbeit wurden beide Ansätze in Kombination verwendet. Es wurden also 2 verschiedene Datenquellen herangezogen, was bei der Aufteilung der Datensätze eine wichtige Rolle spielt (siehe Kapitel Einteilung der Datensätze<!--TODO: REF -->). 

#### FER+

Als großer frei Verfügbarer Datensatz wurde der *Facial Expression Recognition+* (FER+)[@Barsoum2016] Datensatz verwendet. Bei den Eingangsdaten des *FER+* handelt es sich um dieselben Bilder, wie auch beim *FER2013*, welcher Teil der International Conference for Machine Learning (ICML) Challenge 2013 war und danach der Öffentlichkeit zur Verfügung gestellt wurde. Bei FER+ wurden jedoch alle *Label* mithilfe von *Crowdsourcing* neu erstellt, um eine bessere Datenqualität zu erreichen (vgl. [@Barsoum2016]). Der Datensatz besteht aus 34034 48x48 Graustufen Bilder von Gesichtern. Jedes dieser Bilder wurde von je 10 Freiwilligen mithilfe von *Crowdsourcing* bewertet. Der Datensatz enthält für jede Klasse (Emotionen des *FACS* (inkl. neutral), "kein Gesicht" und "unbekannt" ) die Anzahl an Freiwilligen, welche das Bild entsprechend bewertet haben.
Ein Beispiel für ein einzelnes Datum des Datensatzes ist in Abbildung \ref{single_ferplus} zu sehen.

TODO: Abbildung FER+ Single row image

Das Team von Microsoft Research [@Barsoum2016] beschreibt mehrere Variationen wie die mehrfach *gelabelten* Daten verwendbar sind. In dieser Arbeit wird jedoch ausschließlich der einfache Mehrheits-Ansatz verfolgt. Es wird also jedes Bild der Klasse zugeordnet, welche die meisten Stimmen erhalten hat.

Zum Laden der Daten wurde das folgende Python Skript verwendet.

```python
def load_data_ferplus(fer_ds_path = "fer+/fer2013/fer2013.csv", ferplus_ds_path="fer+/fer2013new.csv"):
    # loading raw data
    ## loading only the label cols from fer+
    cols=['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    ferplus = pd.read_csv(ferplus_ds_path , usecols=cols, dtype=np.int32)
    # and only the pcitures (pixels) from original fer2013
    fer = pd.read_csv(fer_ds_path, usecols=['pixels'])
    # getting simple majority voted label using idxmax and merge it with the picture dataset
    merged_fer = pd.concat([fer, ferplus.idxmax(axis=1)], axis=1)
    # redefining column names
    merged_fer.columns = ['img_pixels', 'emotion']
    return merged_fer
```

#### selbsterstellte Daten

Zur Generierung von eigenen Daten wurde im Rahmen der Arbeit eine einfache Website erstellt, welche mithilfe von *WebRTC* Zugriff auf die Kamera bekommt. Auf dieser Website haben freiwillige Probanden und auch der Autor dieser Arbeit die Möglichkeit, nacheinander für jede Emotion des *FACS* ein Video in der Länge von 15 Sekunden aufzunehmen. Dieses wird anschließend direkt auf dem Server gespeichert. Die Webseite ist in Abbildung \ref{webrtc_screenshot} zu sehen.

![Screenshot der Video-Recording Website\label{webrtc_screenshot}](source/figures/web_recorder.png){ width=70% }

Mithilfe dieser Webseite wurden insgesamt 20 Sätze von 8 verschiedenen freiwilligen Probanden (den Autor dieser Arbeit eingeschlossen) gesammelt. Aus diesen Videos wurde anschließend mit Hilfe des folgenden Python-Skripts pro Sekunde ein Einzelbild extrahiert und mit dem Namen der entsprechenden Klasse abgespeichert. Somit wurden also $20 * 15 = 300$ Einzelbilder pro Klasse generiert. 

```python
def extract_video_frames(prefix, videofile,
	targetdir = "../data/extracted"):

    vidcap = cv2.VideoCapture(videofile)
	fps = min((50, int(vidcap.get((cv2.CAP_PROP_FPS)))))
	print ("extracting every {} frame from {}".format(fps,videofile))
    success,image = vidcap.read()
    count = 0
    while success:
        if (count % fps) == 0:
			# save frame as JPEG file
            cv2.imwrite("{}/{}_frame{}.jpg".format(targetdir,
				prefix , count) , image)  
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
```
<!-- Python-Skript zum extrahieren der Einzelbilder\label{listing_extract_video} -->

Die Einzelbilder wurden anschließend manuell auf Korrektheit, das heißt Zuordnung zur Klasse, geprüft. Dabei wurden insgesamt 402 Bilder wieder aussortiert (TODO: Verify numbers).

## Datenpräparation

Um die Effizienz, sowie die Genauigkeit der Vorhersage des neuronalen Netzwerkes zu steigern, werden alle Daten bevor sie dem KNN präsentiert werden auf diverse Arten präpariert. Im Folgenden wird auf die angewandten Methoden genauer eingegangen.

### Vorverarbeitung

Um ein möglichst gutes Ergebnis zu erzielen, wurden die selbst erstellten Daten und die frei verfügbaren Daten aus dem *FER+* Datensatz auf diverse Weise vorverarbeitet. 
Bei den selbst aufgezeichneten Daten wurde, wie bereits erwähnt, eine abschließende manuelle Sichtung vorgenommen, um möglichst alle falsch markierten Daten auszusortieren. Des Weiteren werden alle selbst aufgezeichneten Bilder auf den Ausschnitt des Gesichtes beschränkt, bevor sie dem selbst entworfenen KNN präsentiert werden. Dazu wird der bereits beschriebene *Viola-Jones-Detektor* [@Shen1997] verwendet. Der das Gesicht enthaltende Teil des Bildes wird anschließend auf 48x48 Pixel skaliert und in ein Graustufenbild umgewandelt. Damit haben letzten Endes alle Eingangsdaten des neuronalen Netzes die gleiche Struktur. Der Python Code für diese Vorverarbeitung ist im folgenden Listing zu sehen.

TODO: Listing, preprocess selfrecorded data.

Bei den Daten aus dem *FER+* Datensatz ist etwas weniger Vorverarbeitung nötig. Die Vorverarbeitung dieser Daten besteht im wesentlichen darin, die Pixelwerte, welche als ein eindimensionales Array vorliegen, in eine 48x48 Matrix umzuwandeln. Des Weiteren wird aus den mehrstimmigen *Labels* des *FER+* Datensatzes mithilfe des einfachen Mehrheitsprinzips das hier Genutzte extrahiert. In einem letzten Schritt werden alle Datensätze, welche mit "not a face" oder "unknown" markiert sind, aussortiert.

```python
def prep_data_ferplus(ferplus_data):
    # removing NF (no face) and unknown columns
    cleaned = ferplus_data[ferplus_data['emotion'] != "NF"][ferplus_data['emotion'] != "unknown"]
    ### Getting oneHot encoded classes using get_dummies
    emotions = pd.get_dummies(cleaned['emotion']).as_matrix()
    ## retrieving images from pixel list
    img_pixels = cleaned['img_pixels'].tolist()
    img_width, img_height = 48, 48
    imgs = []
    # extracting images from space delimited pixel values
    for pixel_row in img_pixels:
        img = [int(pixel) for pixel in pixel_row.split(' ')]
        #Having them in a one-dim list now, we have to make an np array with our img_shape
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

Je mehr Trainingsdaten für das KNN vorhanden sind, desto besser kann es auch mit ungesehenen Daten umgehen. Da nur begrenzt viele Daten zur Verfügung stehen werden, in dieser Arbeit einige Methoden der künstlichen  Datenvermehrung angewandt. Dazu werden die Bilder der Eingangsdaten zum Beispiel gespiegelt, verzerrt oder gedreht. In dieser Arbeit wurden die Methoden der Spiegelung, sowie des zufälligen Drehens einiger Bilder angewandt. Durch die Spiegelung, bzw. Drehung eines Bildes entsteht wieder ein neues Bild, welches zum Training des neuronalen Netzes verwendet werden kann. So kann mit dieser relativ einfachen Methode die Anzahl der Trainingsdaten sehr leicht verdoppelt werden. Zur Datenmehrung während des Trainingsprozesses wurde der *ImageDataGenerator* aus dem *keras* Modul verwendet. Dieser bietet die Möglichkeit zufällige Bilder zu spiegeln oder zu rotieren. Dazu wurden die Parameter *rotation_range* und *horizontal_flip* entsprechend gesetzt.

```python
img_gen = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=45,
                            horizontal_flip=True)
```

### Einteilung der Datensätze

Beim maschinellen Lernen ist es üblich den vorhandenen Datensatz bzw. die vorhandenen Datensätze in verschiedene Verwendungszwecke einzuteilen. Klassisch spricht man hier vom *train/test-Split*, also einer Aufteilung der Daten in einen Trainings- und einen Test-Datensatz. In modernen Projekten, welche sich mit maschinellen Lernen beschäftigen, spricht man jedoch zumeist von einem *train/dev/test-split*. Die Daten werden also in einen Trainings-, einen Entwicklungs- und einen Test-Datensatz eingeteilt. Als Entwicklungs-Datensatz bezeichnet man jene Daten, welche während der Entwicklung, also dem Anpassen bestimmter (Hyper-)Parameter, des neuronalen Netzes zur Evaluierung verwendet werden. Der Test-Datensatz ist in diesem Szenario ein Satz aus Daten, welches das neuronale Netz vor der Fertigstellung noch nicht "zu sehen" bekommen hat. Beim klassischen *train/test-split* ist der Test-Satz also eigentlich das, was wir heute als Entwicklungs-Datensatz bezeichnen und es gibt keinen wirklichen Test-Datensatz.
Bei der Wahl der Datenquellen ist es wichtig, dass die Test-Daten möglichst ähnlich zu den später erwarteten Eingangsdaten sind und die Entwicklungs- und Test-Datensatz aus der selben Quelle stammen.

Für diese Arbeit bedeutet das, dass die Entwicklungs- und Test-Daten aus den selbstgenerierten Daten stammen, da diese bereits von aufgenommenen Videos stammen, was den Zeildaten sehr nahe kommt.
Als Trainingsdaten wird entsprechend der *FER+* Datensatz verwendet.
Ein Problem bei einer solchen Aufteilung, wenn also die Trainingsdaten aus einem anderen Datensatz stammen als die Entwicklungs und Test-Daten, ist, dass man gewisse Probleme, wie zum Beispiel eine Überanpassung, teilweise nur schwer erkennen kann. Deshalb ist es in einem solchen Fall sinnvoll noch einen vierten Datensatz einzuführen, welcher aus der selben Quelle wie die Trainingsdaten stammt (hier *FER+*). Man spricht hier vom *dev_train* oder auch *bridge* Datensatz. Dieser wird im Prinzip analog zum Entwicklungsdatensatz behandelt und dient zum Testen der Parameter des neuronalen Netzwerkes nach jeder Änderung. Anhand der Unterschiedlichen Ergebnisse für den *bridge* und den *dev* Datensatz kann man nun schnell bestimmte Probleme des neuronalen Netzwerks erkennen.

In dieser Arbeit wurde daher auch die Einteilung in 4 Datensätze gewählt. Die Daten wurden dabei wie folgt aufgeteilt:
Die selbsterstellten Daten wurden in zu je 50% in den Entwicklungs- und Test-Datensatz aufgeteilt. Vom *FER+* Datensatz wurden 10% der Bilder für den *Bridge* Datensatz verwendet und 90% als Trainingsdaten. Die Aufteilung ist in Abbildung \ref{data_split} veranschaulicht.

![Aufteilung der Daten in 4 Datensätze \label{data_split}](source/figures/train_test_split.pdf){ width=90% } <!-- TODO: enter number -->

Zum Aufteilen der einzelnen Datensätze wurde die Funktion "train_test_split" aus dem Python Modul "sklearn" <!--TODO: ref --> verwendet. Um eine zwar anfangs zufällige, jedoch reproduzierbare Aufteilung zu erhalten, wird der "Random_state" auf einen festen Wert gesetzt. Das genutzte Python Skript ist im Folgenden abgebildet.

```python
def split_datasets(ferplus_imgs, ferplus_emotions, selfrecorded_imgs, selfrecorded_emotions):
    xTrain, xBridge, yTrain, yBridge = train_test_split(
        ferplus_imgs, ferplus_emotions, test_size = 0.1, random_state = 20808)
    xDev, xTest, yDev, yTest = train_test_split(
        ferplus_imgs, ferplus_emotions, test_size = 0.5, random_state = 280919)
    return xTrain, xBridge, yTrain, yBridge, xDev, xTest, yDev, yTest
```

## Entwurf und Entwicklung neuronaler Netze

Im Rahmen dieser Arbeit sollen 3 verschiedene neuronale Netzwerke entworfen, trainiert und evaluiert werden. Alle 3 sollen gefaltete neuronale Netze sein, sich jedoch in der Topologie und den (Hyper-) Parametern unterscheiden.

### Entwicklungsumgebung

Die Entwicklung der KNN´s wurde in Python mithilfe der Machine-Learning Erweiterunge Keras vorgenommen. Keras ist eine vereinfachte Schnittstellenimplementierung zur einfachen Verwendung von verschiedenen Machine Learning-Schnittstellen. In dieser Arbeit wurde Keras mit der Tensorflow-Schnittstelle verwendet. 
Als Entwicklungsumgebung wurde hierzu ein Jupyter Notebook verwendet. Der Vorteil eines Jupyter Notebook liegt darin, dass sehr einfach Text und Programmierabschnitte, sowie deren Ausgabe nebeneinander visualisiert werden können.

### Topologie

Unter der Topologie des KNN versteht man die Architektur im Zusammenhang mit den (Hyper-)Parametern. Die Architektur beschreibt den Aufbau oder die Struktur des Netzwerkes. In einem CNN also im wesentlichen die Art und Reihenfolge der einzelnen Netzwerkschichten.

Als Hyperparameter hingegen bezeichnet man weitere Rahmenparameter, welche unabhängig vom grundlegenden Aufbau des Netzes verändert werden können. Einige solcher Hyperparameter wurden in Kapitel 2 bereits vorgestellt.

In dieser Arbeit sollen drei verschiedene Netzwerktopologien für das Problem der Emotions-Klassifizierung entworfen werden. 

#### Einfaches faltendes neuronales Netz

Als erstes Modell soll ein sehr einfaches faltendes neuronales Netz entworfen werden. 

##### Architektur

Das Neuronale Netzwerk besteht aus insgesamt 4 *Faltungs-Stapeln*, gefolgt von einer Ausgabeschicht. Die *Faltungs-Stapel* bestehen jeweils aus zwei Faltungsschichten, auf welchen, abgesehen von der letzten Schicht, eine Stapel-Normalisierung (engl. batch normalization) angewandt wird, gefolgt von einer Pooling Schicht, auf welche eine *Dropout*-Regularisierung angewandt wird. In Abbildung \ref{architecture_simple_cnn} ist die Architektur kurz dargestellt.
TODO: Abbildung Architektur \label{architecture_simple_cnn}

##### Hyper-Parameter

Die Topologie des KNN lässt sich durch die folgend beschriebenen Parameter konkretisieren.

* **Filteranzahl**: Die Anzahl der Filter in den einzelnen Faltungsschichten beeinflusst die Dimension der Folgedaten. Für dieses Netzwerk wurde die Filteranzahl für beide Faltungsschichten eines Stapels in der Regel gleich gesetzt. Mit jedem Stapel verdoppelt sich die Anzahl der Filter. Lediglich die letzte Faltungsschicht weicht von diesem Schema ab und verwendet die fixe Anzahl von 8 Filtern, was der Anzahl der Klassen $|Z|$ entspricht. Die Anzahl der Filter der ersten Faltungsschicht wurde auf 16 festgelegt. Die Schichten in den darauffolgenden Stapeln verwenden also jeweils 32, 64, und 128 Filter.

* **Filtergröße**: Auch die Filtergröße hat einen nicht zu vernachlässigenden Einfluss auf die Dimension der Daten. In diesem Netzwerk wurde für den ersten Stapel eine Filtergröße von $7x7$ festgelegt. Für die 2 folgenden Stapel wurde die Filtergröße jeweils um $2x2$ verringert und beträgt damit $5x5$, bzw. $3x3$. Der letzte Stapel verwendet ebenfalls eine Filtergöße von $3x3$.

* **Füllung**: Die Füllung (engl. padding) bei einer *convolutional* Schicht beeinflusst im Zusammenspiel mit der Filtergröße die Dimension der Ausgabedaten. In diesem Modell wurde für jede Faltungsschicht das sogenannte *same padding* verwendet. Das bedeutet, dass genau so viele Zeilen aufgefüllt werden, die nötig sind, damit die zweidmensionale Ausgabedimension der Eingabedimension entspricht. Da sich im Allgemeinen die Dimension der Ausgabe einer neuronalen Faltungsschicht mit einer Eingabedimension von $n x n$ mit einem *padding* von $p$ und einer Filtergröße von $f x f$ wie folgt berechnen lässt. 
$$
conv(n x n, f x f, p) = n +2p - f + 1 x n +2p - f + 1
$$
Ergibt sich für das *same padding* die Größe der Füllung $p$ wie folgt.

$$
n + 2p_{same} -f +1 = n
$$
$$
p_{same} = \frac{f-1}{2}
$$

* **Pooling**: Als Pooling Methode wird in diesem Netzwerk das Durchschnitts-Pooling (engl. *average pooling*) angewandt. Beim Average Pooling, wird ähnlich wie bei einer *convolutional* Schicht, ein Filter einer definierten Größe (hier $2x2$) über die Daten *geschoben*. Beim *average pooling* erhält Das Ziel-Fenster immer den Durchschnittswert, der Werte im Fenster (Vergleich Abbildung \ref{avg_pooling}). Die letzte *pooling* Schicht in diesem Netz stellt eine Besonderheit dar, da hier da sogenannte globale Durchschnitts-Pooling verwendet wurde. Beim *global average pooling* wird für jede zweidmensionale Ebene genau ein Durchschnittswert gebildet, die Filtergröße ist also gleich der Größe der Eingangsdaten.

TODO: Abbilding AVG Pooling \label{avg_pooling}

* **Kosten-Funktion**: Zur Ermittlung des Netzwerksfehlers wird die kategorische Kreuzentropie-Funktion verwendet.<!-- TODO: ref --> Auf die genaue Funktionsweise wird in dieser Arbeit nicht genauer eingegangen.

#### abgewandeltes XCeption Net

#### zeitabhängiges faltendes neuronales Netz


### Hyperparmater


## Evaluierung neuronaler Netze?

### abschnitt 1

### abschnitt 2

## Entwicklung eines Webservice

Nachdem sich für das passende neuronale Netz entschieden wurde wird dieses einem Webservice zur Verfügung gestellt. Der Webservice hat die Aufgabe Videodaten entgegen zu nehmen und sekundenweise Einzelbilder an den Klassifizierer zu übergeben und anhand der Ausgabe eine Zeitleiste mit den erkannten Emotionen zurückliefen.

### Softwarearchitektur 

Der Webservice wurde mithilfe der Python Erweiterung Flask realisiert.
Flask ist eine schlanke Erweiterung zur einfachen Erstellung von Web Diensten in Python. Der Werbservice, sowie der Klassifizierer sind als sogenannte Microservices aufgebaut welche in separaten *Containern* laufen. (siehe Abbildung \ref{app_architecture})

TODO: Abbildung Architektur \label{app_architecture}





<!-- expose:
3. Umsetzung & Evaluierung
	1. Prototyping
		1. Aufbau verschiedener neuronaler Netze
		2. Trainieren der Modelle
		3. Bereitstellung als Webservice
	2. Experiment
		1. Verifizieren mit Testdaten
		2. Untersuchung der Genauigkeit
	3. Ergebnisse
-->