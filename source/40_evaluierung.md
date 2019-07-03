\clearpage

# Evaluierung

Dieses Kapitel befasst sich mit der weiterführenden Evaluierung des erstellten KNN. Ein großteil der Evaluierung wurde bereits während der Implementierung vorgenommen um möglichst gute Hyperparameter für das Netzwerk zu finden. Ziel dieses Kapitels soll es sein aus dem gewählten Model noch weitere Erkenntnisse zu gewinnen und die Ergebnisse zu visualisieren.

## abschließende Bewertung der Netzwerkleistung

Nach der finalen Implementierung des neuronalen Netzwerkes wurde ein Letzter Validierungslauf für den *Bridge*, das Entwicklungs sowie auch für den Test-Datensatz gestartet. Dabei wurden die Erkenntnisse aus TODO: Kapitel 3 bestätigt, dass der Klassifizierer weniger gut mit nicht erlenten Daten umgehen kann. Tabelle \ref{table_acc} zeogt die Genauigkeit der einzelnen Datensätze während des Validierungslaufes. Wie zu erwarten schneidet der Test-Datensatz ähnlich ab wie der Entwicklungsdatensatz.

Datensatz | Genauigkeit |
| ----- | ----: |
| Trainings-Daten | 62,3% <!--TODO: verify --> |
| Bridge-Daten | 45,2% |
| Entwicklungs-Daten | 29,0% | 
| Test-Daten | 29,6,2% |

Bei einem Mehrklassen-Klassifizierer, wie er in dieser Arbeit entwickelt wurde, bietet sich auch immer eine Darstellung einer Verwirrungs-Matrix an. Diese stellt dar, wie oft eine Klasse fälschlicherweise als eine andere klassifiziert wurde. in Abbildung \ref{conf_matrix_bridge} und \ref{conf_matrix_test} ist die Verwirrung-Matrix für den *Bridge* sowie den Test Datensatz auf dem erstellten Model zu sehen.

![Verwirrungs-Matrix für den *Bridge* Datensatz. Auf der Y-Achse sind die tatsächlichen Emotionen zu sehen, auf der X-Achse die vom Netz vorhergesagten Emotionen.  \label{conf_matrix_bridge}](source/figures/conf_matrix_bridge.png){ width=80% }

![Verwirrungs-Matrix für den Test Datensatz. \label{conf_matrix_test}](source/figures/conf_matrix_test.png){ width=80% }

In den beiden Verwirrungs-Matrizen kann man sehen, dass die Emotionen *Verachtung*, *Ekel*, *Angst* und *Traurigkeit*, bis auf wenige kleine Ausnahmen immer falsch zugewiesen wurden. Die meisten falsch zugewiesenen Emotionen wurden fälschlicherweise als Wut klassifiziert.

Dies deutet wieder auf unklar definierte Trainingsdaten, oder einfach zu wenige Trainingdaten hin. Eine zukünftige Verbesserung könnte durch eine genauere Prüfung der Trainingsdaten erreicht werden.

## Testen des Webservice mit Fremddaten

Abschließend wurde der erstellte Webservice getestet. Dazu wurde ein Satz an Videos eines Probanden, der nicht im selbsterstellten Datensatz zu finden war an den Webservice gesendet und das Ergebnis manuell ausgewertet.
Zur Aufnahme des Videos wurde derselbe Webservice verwendet, wie auch schon zur Beschaffung der Entwicklungs-/Test-Daten. Das Video wurde mithilfe der minimalsitischen Benutzerschnittstelle an den Webservice gesendet. Das Ergebnis eines Beispiel Videos (für die Emotion Freude) ist in Abbildung \ref{screenshot_testweb} zu sehen.

![Verwirrungs-Matrix für den Test Datensatz. \label{screenshot_testweb}](source/figures/conf_matrix_test.png){ width=80% }
```JSON
{"emotions":[{"emotion":"happiness","time":0},{"emotion":"happiness","time":1},{"emotion":"no_face","time":2},{"emotion":"no_face","time":3},{"emotion":"no_face","time":4},{"emotion":"happiness","time":5},{"emotion":"happiness","time":6},{"emotion":"happiness","time":7},{"emotion":"happiness","time":8},{"emotion":"happiness","time":9},{"emotion":"happiness","time":10},{"emotion":"happiness","time":11},{"emotion":"happiness","time":12},{"emotion":"happiness","time":13}],"videoname":"happiness_55zc2lbb1o7y5s3y37.webm"}
```


