\clearpage

# Umsetzung & Evaluierung

In diesem Teil wird die Umsetzung der Arbeit eingegangen. im ersten Abschnitt wird auf die Art und Einteilung der zu klassifizierenden Daten eingegangen. Im Weiteren Verlauf wird die Beschaffung der verschiedenen Datensätze behandelt mit welchen gearbeitet werden soll. Anschließend werden auf diverse Verfahren der Datenpräparation wie Vorbereitung, Normalisierung und Datenvervielfältigung thematisiert. Letzten Endes werden verschiedene neuronale Netze entworfen, trainiert und getestet, bis dann im letzten Abschnitt die Realisierung des zu erstellenden Webservice näher betrachtet wird.

## Beschaffenheit und Einteilung der Daten

Wie bereits beschrieben, sollen für den späteren Webservice Videodaten als Eingabe dienen. Das neuronale Netz wird jedoch nicht direkt die Videodaten, sondern aus dem Video extrahierte Einzelbilder als Eingabe erhalten. Diese Eingabedaten werden anhand später beschriebener Verfahren noch weiter vorbereitet und optimiert.
Die Einzelbilder sollen als *Array* von Pixelwerten eines Graustufenbildes an das neuronale Netzwerk übergeben werden, siehe Kapitel Datenpräparation <!-- TODO: REF -->.
Die Bilder sollen in 7 verschiedene Klassen. Die verschiedenen Emotionen bilden den Zielklassenvektor $Z$, es gilt also $|Z| = 7$. Jedes Bild welches der selben Emotion des *FACS* entspricht ist Mitglied der selben Klasse aus $Z$.

## Datensätze

Ein Teil der Arbeit bestand darin geeignete Datensätze für das Training und die Evaluierung des neuronalen Netzes zu finden und diese später in entsprechende Trainings, Evaluierungs und Test-Datensätze für das neuronale Netzwerk zu unterteilen.

### Beschaffung der Datensätze

Dazu sollen 2 verschiedene Ansätze unterschieden werden. Zum einen die Beschaffung eines vorhandenen freien Datensatzes von Gesichtsbildern inklusive der Zuordnung zu einer der entsprechenden *FACS* Emotionen, sowie zum anderen die Generierung von eigenen Daten mithilfe eines Webservice und freiwillgen Probanden. 
In dieser Arbeit wurden beide Ansätze in Kombination verwendet. Es wurden also 2 verschiedene Datenquellen herangezogen, was bei der Aufteilung der Datensätze noch eine wichtige Rolle spielt (siehe Kapitel Einteilung der Datensätze<!--TODO: REF -->). 

#### FER+

Als großer frei Verfügbarer Datensatz wurde der *Facial Expression Recognition+* (FER+)[@Barsoum2016] Datensatz verwendet. Bei den Eingangsdaten des *FER+* handelt es sich um die selben Bilder, wie auch beim *FER2013*, welcher Teil der International Conference for Machine Learning (ICML) Challenge 2013 war, und danach der Öffentlichkeit zur Verfügung gestellt wurde. Bei FER+ wurden jedoch alle *Label*, mithilfe von *Crowdsourcing* neu erstellt, um eine bessere Datenqualität zu erreichen. (vgl. [@Barsoum2016]). Der Datensatz besteht aus 34034 48x48 Graustufen Bilder von Gesichtern. Jedes dieser Bilder von je 10 Freiwilligen mithilfe von *Crowdsourcing* bewertet. Der Datensatz enthält für jede Klasse (Emotionen des *FACS* und "kein Gesicht") die Anzahl an Freiwilligen, welche das Bild entsprechend bewertet haben.
Ein Beispiel für ein einzelnes Datum des Datensatzes ist in Abbildung \ref{single_ferplus} zu sehen

TODO: Abbildung FER+ Single row image

Das Team von Microsoft Research [@Barsoum2016] beschreibt mehrere Variationen, wie die mehrfach *gelabelten* Daten verwendbar sind. In dieser Arbeit wird jedoch ausschließlich der einfache Mehrheits-Ansatz verfolgt. Es wird also jedes Bild der Klasse zugeordnet, welche die meisten Stimmen erhalten hat.

#### selbsterstellte Daten

Zum selbst erstellen von Daten wurde im Rahmen der Arbeit eine einfache Website erstellt, welche mithilfe von *WebRTC* Zugriff auf die Kamera bekommt. Auf dieser Website haben freiwillige Probanden, und auch der Autor dieser Arbeit, die Möglichkeit nacheinander für jede Emotion des *FACS* ein 15 sekündiges Video aufzunehmen. Dieses wird anschließend direkt auf dem Server gespeichert. Die Webseite ist in Abbildung \ref{webrtc_screenshot} zu sehen.

![Screenshot der Video-Recording Website\label{webrtc_screenshot}](source/figures/web_recorder.png){ width=70% }

Mithilfe dieser Webseite wurden insgesamt 15 Sätze an 15 sekündigen Videos von 8 verschiedenen freiwilligen Probanden (den Autor dieser Arbeit eingeschlossen) gesammelt. Aus diesen Videos wurde anschließend mit Hilfe des folgenden Python-Skripts pro Sekunde ein Einzelbild extrahiert, und mit dem Namen der entsprechenden Klasse abgespeichert. Somit wurden also $15 * 15 = 225$ Einzelbilder pro Klasse generiert. 

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

Die Einzelbilder wurden anschließend manuell auf Korrektheit, das heißt Zuordnung zur Klasse, geprüft. Dabei wurden insgesamt 200 Bilder wieder aussortiert (TODO: Verify numbers).


### Einteilung der Datensätze

Beim maschinellen Lernen ist es üblich den vorhandenen Datensatz, bzw. die vorhandenen Datensätze in verschiedene Verwendungszwecke einzuteilen. Klassisch spricht man hier immer vom *train/test-Split*, also einer Aufteilung der Daten in einen Trainings- und einen Test-Datensatz. In modernen Projekten, welche sich mit maschinellen Lernen beschäftigen spricht man jedoch zumeist von einem *train/dev/test-split*. Die Daten werden also in einen Trainings-, einen Entwicklungs- und einen Test-Datensatz eingeteilt. Als Entwicklungs-Datensatz bezeichnet man, jene Daten, welche während der Entwicklung, also dem Anpassen bestimmter (Hyper-)Parameter, des neuronalen Netzes zur Evaluierung verwendet werden. Der Test-Datensatz ist in diesem Szenario ein Satz aus Daten, welches das neuronale Netz vor der Fertigstellung noch nicht "zu sehen" bekommen hat. Beim klassischen *train/test-split* ist der Test-Satz also eigentlich, das was wir Heute als Entwicklungs-Datensatz bezeichnen, und es gibt keinen wirklichen Test-Datensatz.
Bei der Wahl der Datenquellen, ist es wichtig, dass die Test-Daten möglichst ähnlich, zu den später erwarteten Eingangsdaten sind, und dass Entwicklungs- und Test-Datensatz aus der selben Quelle stammen sollten.

Für diese Arbeit bedeutet das, dass die Entwicklungs- und Test-Daten aus den selbstgenerierten Daten stammen, da diese bereits von aufgenommenen Videos stammen, was den Zeildaten sehr nahe kommt.
Als Trainingsdaten wird entsprechend der *FER+* Datensatz verwendet.
Ein Problem bei einer solchen Aufteilung, wenn also die Trainingsdaten aus einem anderen Datensatz stammen als die Entwicklungs und Test-Daten, ist, dass man gewisse Probleme, wie zum Beispiel eine Überanpassung teilweise nur schwer erkennen kann. Deshalb ist es in einem solchen Fall sinnvoll noch einen vierten Datensatz einzuführen, welcher aus der selben Quelle wie die Trainingsdaten stammt (hier *FER+*). Man spricht hier vom *dev_train* oder auch *bridge* Datensatz. Dieser wird im Prinzip analog zum Entwicklungsdatensatz behandelt und dient zum Testen der Parameter des neuronalen Netzwerkes nach jeder Änderung. Anhand der Unterschiedlichen Ergebnisse für den *bridge* und den *dev* Datensatz kann man nun schnell bestimmte Probleme des neuronalen Netzwerks erkennen.

In dieser Arbeit wurde daher auch die Einteilung in 4 Datensätze gewählt. Die Daten wurden dabei wie folgt aufgeteilt:
Die selbsterstellten Daten wurden in zu je 50% in den Entwicklungs- und Test-Datensatz aufgeteilt. Vom *FER+* Datensatz wurden  10% Der Bilder für den *Bridge* Datensatz verwendet und 90% als Trainingsdaten. Die Aufteilung ist in Abbildung \ref{data_split} veranschaulicht.

![Aufteilung der Daten in 4 Datensätze \label{data_split}](source/figures/train_test_split.pdf){ width=90% } <!-- TODO: enter number -->

 <!--like this https://www.freecodecamp.org/news/what-to-do-when-your-training-and-testing-data-come-from-different-distributions-d89674c6ecd8/ -->

Um eine Eingangs zufällige, jedoch immer gleich reproduzierbare Aufteilung zu erzielen wurde das folgende Python Skript verwendet.

TODO: Code Listing Data_split.py


## Datenpräparation

Um die Effizienz, sowie die Genauigkeit der Vorhersage des neuronalen Netzwerkes zu steigern werden alle Daten, bevor Sie dem KNN präsentiert werden auf diverse Arten präpariert. Im Folgenden wird auf die angewandten Methoden genauer eingegangen.

### Vorverarbeitung

Um ein möglichst gutes Ergebnis zu erzielen wurden die selbst erstellten Daten, aber auch die frei verfügbaren Daten aus dem *FER+* Datensatz auf diverse Weise vorverarbeitet. 
Bei den selbst aufgezeichneten Daten, wurde wie bereits erwähnt eine abschließende manuelle Sichtung vorgenommen, um möglichst alle falsch markierten Daten auszusortieren. Des weiteren werden alle der selbst aufgezeichneten Bilder auf den Ausschnitt des Gesichtes beschränkt, bevor sie dem selbst entworfenen KNN präsentiert werden. Dazu wird der bereits beschriebene *Viola-Jones-Detektor* [@Shen1997] verwendet um das Gesicht im jeweiligen Bild zu detektieren. Nachdem der, das Gesicht enthaltende, Teil des Bildes ausgeschnitten wurde, wird dieser auf 48x48 Pixel skaliert und in ein Graustufenbild umgewandelt. Damit haben letzten Endes alle Eingangsdaten des neuronalen Netzes die gleiche Struktur. Der Python Code für diese Vorverarbeitung ist im folgenden Listing zu sehen.

TODO: Listing, preprocess selfrecorded data.

Bei den Daten aus dem *FER+* Datensatz ist etwas weniger Vorverarbeitung nötig. Die Vorverarbeitung dieser Daten besteht im wesentlichen darin, die Pixelwerte welche als ein eindimensionales Array vorliegen in eine 48x48 Matrix umzuwandeln. Des weiteren wird aus den mehrstimmigen *Labels* des *FER+* Datensatzes mithilfe des einfachen Mehrheitsprinzips das hier genutzte extrahiert. In einem letzten Schritt werden dann alle Datensätze, welche mit "not a face" markiert sind, aussortiert. (siehe Listing TODO:) 

TODO: CODELISTING PROCESS FER+


### Normalisierung

*"Data normalization has been proposed to address the aforementioned challenge by reducing the training space and making the
training more efficient."* [@Zhang2018]
<!--
Laut [@Zhang2018] hilft die Normalisierung also dabei die Herausforderung einer effizienten Erkennung zu meistern, indem das Trainingsspektrum verkleinert wird, was die Trainingsphase beschleunigt.-->

Ein üblicher Schritt, um die Trainingsphase  im maschinellen Lernen zu beschleunigen ist es die Eingabedaten zu normalisieren. Ziel ist es die Eingabedaten, welche auf einem sehr breiten Spektrum liegen so zu normalisieren um das Spektrum zu verkleinern. Im vorliegenden Fall geht es um die Graustufenbilder. Im Generellen kann man Normalisierung von solchen Bilden wie folgt beschreiben: Ein n-dimensionales Graustufenbild $I:\{\mathbb{X}\subseteq\mathbb{R}^n\}\rightarrow\{\text{Min},..,\text{Max}\}$ mit den Pixelwerten zwischen $Min$ unx $Max$ wird in ein neues Graustufenbild $I_N:\{\mathbb{X}\subseteq\mathbb{R}^n\}\rightarrow\{\text{newMin},..,\text{newMax}\}$ mit Pixelwerten zwischen $newMin$ und $newMax$ überführt.[@gonzalez2008digital]
Die lineare Normalisierung eines Graustufenbildes berechnet sich wie folgt:

$$
I_N=(I-\text{Min})\frac{\text{newMax}-\text{newMin}}{\text{Max}-\text{Min}}+\text{newMin}
$$

Im vorliegenden Beispiel sind die Ausgangswerte für $Min = 0$ und $Max = 255$ und für eine einfache Normalisierung werden die Werte $newMin = 0$ und $newMax = 1$ gewählt, damit alle Datenwerte zwischen 0 und 1 liegen. Damit ergibt sich die vereinfachte Formel:

$$
I_N=\frac{I}{\text{Max}} \Rightarrow I_N=\frac{I}{255}
$$

Zur Normalisierung der Daten werden dementsprechend alle Pixelwerte durch 255 dividiert, bevor das Bild dem neuronalen Netz gezeigt wird.

### Datenmehrung

Je mehr Trainingsdaten für das KNN vorhanden sind, desto besser kann es auch mit ungesehenen Daten umgehen. Da nur begrenzt viele Daten zur Verfügung stehen werden in dieser Arbeit einige Methoden der künstlichen  Datenvermehrung angewandt. Dazu werden die Bilder der Eingangsdaten zum Beispiel gespiegelt, verzerrt oder gedreht. In dieser Arbeit wurden die Methoden der Spiegelung, sowie des zufälligen drehens einiger Bilder angewandt. Durch die Spiegelung, bzw. Derhung eines Bildes entsteht wieder ein neues Bild, welches zum Training des neuronalen Netzes verwendet werden kann. So kann mit dieser relativ einfachen Methode die Anzahl der Trainingsdaten sehr einfach verdoppelt werden. Die Methode die zur Vermehrung der Bilder des *FER+* Datensatzes verwendet wurde ist nachfolgende abgebildet.

TODO: Code Listing spiegeln/drehen

## Entwurf und Entwicklung neuronaler Netze

Im Rahmen dieser Arbeit sollen 3 verschiedene neuronale Netzwerke entworfen, trainiert und evaluiert werden. Alle 3 sollen gefaltete neuronale Netze sein, sich jedoch in der Topologie und den (Hyper-) Parametern unterscheiden.

### Entwicklungsumgebung

Die Entwicklung der KNN´s wurde in Python mithilfe der Machine-Learning Erweiterunge Keras vorgenommen. Keras ist eine vereinfachte Schnittstellenimplementierung zur einfachen Verwendung von verschiedenen Machine Learning-Schnittstellen. In dieser Arbeit wurde Keras mit der Tensorflow-Schnittstelle verwendet. 
Als Entwicklungsumgebung wurde hierzu ein Jupyter Notebook verwendet. Der Vorteil eines Jupyter Notebook liegt darin, dass sehr einfach Text und Programmierabschnitte, sowie deren Ausgabe nebeneinander visualisiert werden können.

### Topologie

Unter der Topologie des KNN versteht man die Architektur im Zusammenhang mit den (Hyper-)Parametern. Die Architektur beschreibt den Aufbau, oder die Struktur, des Netzwerkes. In einem CNN also im wesentlichen die Art und Reihenfolge der einzelnen Netzwerkschichten.

Als Hyperparameter hingegen bezeichnet man weitere Rahmenparameter, welche unabhängig vom grundlegenden Aufbau des Netzes verändert werden können. Einige solcher Hyperparameter wurden in Kapitel 2 bereits vorgestellt.

In dieser Arbeit sollen drei verschiedene Netzwerktopoligien für das Problem der Emotions-Klassifizierung entworfen werden. 

#### Einfaches faltendes neuronales Netz

Als erstes Model soll ein sehr einfaches faltendes neuronales Netz entworfen werden. Die Architektur des Netzes sieht dabei wie folgt aus:



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