\clearpage

# Evaluierung  \label{chapter_evaluierung}

Dieses Kapitel befasst sich mit der weiterführenden Evaluierung des erstellten KNN. Ein Großteil der Evaluierung wurde bereits während der Optimierung der Hyperparameter vorgenommen (siehe Kapitel \ref{chapter_optimize}), um möglichst gute Hyperparameter für das Netzwerk zu finden. Ziel dieses Kapitels ist es, aus dem gewählten Modell noch weitere Erkenntnisse zu gewinnen und die Ergebnisse zu visualisieren.

## Abschließende Bewertung der Netzwerkleistung

Nach der finalen Implementierung des neuronalen Netzwerkes wurde ein letzter Validierungslauf für den *Bridge*-, den Entwicklungs- sowie für den Test-Datensatz gestartet. Dabei wurden die Erkenntnisse aus Kapitel \ref{chapter_umsetzung} bestätigt, dass der Klassifizierer beim Behandeln von nicht gelernten Daten einige Optimierungsmöglichkeiten bietet. Tabelle 4.1 zeigt die Genauigkeit der einzelnen Datensätze während des Validierungslaufes. Wie zu erwarten schneidet der Test-Datensatz ähnlich ab wie der Entwicklungsdatensatz.

<!-- NICEMAKING -->
\clearpage

Datensatz | Genauigkeit |
| ----- | ----: |
| Trainings-Daten | 65,3%  |
| Bridge-Daten | 45,2% |
| Entwicklungs-Daten | 29,0% |
| Test-Daten | 29,6% |

Tabelle 4.1: Genauigkeit des Netzwerks für die vier verschiedenen Datensätze. \label{table_41} 

Bei einem Mehrklassen-Klassifizierer, wie er in dieser Arbeit entwickelt wurde, bietet sich auch immer eine Darstellung einer Konfusions-Matrix an. Diese zeigt, wie oft eine Klasse fälschlicherweise als eine andere klassifiziert wurde. In Abbildung \ref{conf_matrix_bridge} und \ref{conf_matrix_test} ist die Konfusions-Matrix für den *Bridge*- sowie den Test-Datensatz auf dem erstellten Modell zu sehen.


![Konfusions-Matrix für den *Bridge* Datensatz. Auf der Y-Achse sind die tatsächlichen Emotionen zu sehen, auf der X-Achse die vom Netz vorhergesagten Emotionen.  \label{conf_matrix_bridge}](source/figures/conf_matrix_bridge.png){ width=80% }

![Konfusions-Matrix für den Test Datensatz. \label{conf_matrix_test}](source/figures/conf_matrix_test.png){ width=80% }

In den beiden Konfusions-Matrizen kann man sehen, dass die Emotionen *Verachtung*, *Ekel*, *Angst* und *Traurigkeit*, bis auf wenige kleine Ausnahmen immer falsch zugewiesen wurden. Die meisten falsch zugewiesenen Emotionen wurden fälschlicherweise als Wut klassifiziert.

Dies deutet auf unklar definierte, oder zu wenige Trainingdaten für diese Emotion hin. Eine zukünftige Verbesserung könnte durch eine genauere Prüfung der Trainingsdaten erreicht werden.

<!-- NICEMAKING -->
\clearpage

## Testen des Webservice mit Fremddaten

Abschließend wurde der erstellte Webservice getestet. Dazu ist ein Satz an Videos eines Probanden, der nicht im selbsterstellten Datensatz zu finden war, an den Webservice gesendet und das Ergebnis manuell ausgewertet worden.
Zur Aufnahme des Videos wurde derselbe Webservice verwendet wie auch schon zur Beschaffung der Entwicklungs-/Test-Daten. Das Video ist mithilfe der minimalistischen Benutzerschnittstelle an den Webservice gesendet worden.

![Screenshot der minimalistischen Benutzerschnittstelle beim Test eines Videos für die Emotion Fröhlichkeit. \label{screenshot_testweb}](source/figures/screenshot_web_test.png){ width=80% }

<!-- NICEMAKING -->
\clearpage

Das Ergebnis eines Beispiel-Videos (für die Emotion Freude) ist in Abbildung \ref{screenshot_testweb} zu sehen. Man sieht in dem Textfeld die Rückgabewerte im JSON Format (für komplette Ausgabe siehe Anhang \ref{anhang_websvc_json}) mit der jeweiligen Sekunde und der erkannten Emotion.

Die Auswertung der Ergebnisse bestätigte die bisher erlangten Erkenntnisse, welche auch aus der Konfusions-Matrix hervorgingen. So wurden die Emotion *Fröhlichkeit* und *Überraschung* als einzige vorwiegend richtig erkannt.

