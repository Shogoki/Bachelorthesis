<!--# Abbildungsverzeichnis {.unnumbered}-->

<!--
For me, this was the only drawback of writing in Markdown: it is not possible to add a short caption to figures and tables. This means that the \listoftables and \listoffigures commands will generate lists using the full titles, which is probably isn't what you want. For now, the solution is to create the lists manually, when everything else is finished.
-->

<!--Abbildung 4.1  Ein Beispieö . . .              \hfill{pp}  
<!-- Figure x.x  Short title of the figure . . .         \setcounter{page}{3}     \hfill{pp}   -->
\clearpage
\pagenumbering{roman}

# Abbildungsverzeichnis {.unnumbered}

\renewcommand{\chaptermark}[1]{\markboth{}{\sffamily #1}}
\chaptermark{Abbildungsverzeichnis}
\setcounter{page}{3}

Abbildung \ref{ki_ml_dl} Einordnung in die künstliche Intelligenz \dotfill \pageref{ki_ml_dl} \newline
Abbildung \ref{bio_neuron} Ein biologisches Neuron \dotfill \pageref{bio_neuron} \newline
Abbildung \ref{perzeptron} Perzeptron \dotfill \pageref{perzeptron} \newline
Abbildung \ref{aktfunktionen} Aktivierungsfunktionen \dotfill \pageref{aktfunktionen} \newline
Abbildung \ref{simple_nn} verinfachtes neuronales Netzwerk \dotfill \pageref{simple_nn} \newline
Abbildung \ref{hunde_klassifizierer} beispielhafter Hunde/Katzen-Klassifizierer \dotfill \pageref{hunde_klassifizierer} \newline
Abbildung \ref{CNN} neuronales Faltungsnetzwerk \dotfill \pageref{CNN} \newline
Abbildung \ref{haar_features} Haar-Merkmale \dotfill \pageref{haar_features} \newline
Abbildung \ref{integral_image} Berechnung eines Integralbildes \dotfill \pageref{integral_image} \newline
Abbildung \ref{integral_image_reuse} weiterführende Berechnung eines Integralbildes \dotfill \pageref{integral_image_reuse} \newline
Abbildung \ref{viola_jones_adaboost} modifizierter AdaBoost Algorithmus \dotfill \pageref{viola_jones_adaboost} \newline
Abbildung \ref{cascade_of_classifiers} kaskadierende Klassifizierer \dotfill \pageref{cascade_of_classifiers} \newline
Abbildung \ref{single_ferplus} Datum aus dem FER+ Datensatz \dotfill \pageref{single_ferplus} \newline
Abbildung \ref{webrtc_screenshot} Webservice zum aufzeichnen der Videos \dotfill \pageref{webrtc_screenshot} \newline
Abbildung \ref{data_split} Aufteilung der Datensätze \dotfill \pageref{data_split} \newline
Abbildung \ref{architecture_simple_cnn} Architektur des einfachen Faltungsnetzwerkes \dotfill \pageref{architecture_simple_cnn} \newline
Abbildung \ref{avg_pooling} Durchnitss-Pooling \dotfill \pageref{avg_pooling} \newline
Abbildung \ref{sepconv} separable convolution \dotfill \pageref{sepconv} \newline
Abbildung \ref{xception_architektur} Architektur des Xception Netzwerkes \dotfill \pageref{xception_architektur} \newline
Abbildung \ref{simple_cnn_training} Trainingsverlauf des einfachen Faltungsnetzwerkes \dotfill \pageref{simple_cnn_training} \newline
Abbildung \ref{best_xcepton_training} Trainingsverlauf des besten Models \dotfill \pageref{best_xcepton_training} \newline
Abbildung \ref{more_regularization_acc} Genauigkeit des Netzwerks mit mehr Regularisierung \dotfill \pageref{more_regularization_acc} \newline
Abbildung \ref{more_regularization_loss} Verlust des Netzwerks mit mehr Regularisierung \dotfill \pageref{more_regularization_loss} \newline
Abbildung \ref{app_architecture} Softwarearchitektur des erstellten Webservice \dotfill \pageref{app_architecture} \newline
Abbildung \ref{frontend} Abbildung der minimalsitischen Benutzerschnittstelle \dotfill \pageref{frontend} \newline
Abbildung \ref{conf_matrix_bridge} Verwirrungsmatrix für den Bridge Datensatz \dotfill \pageref{conf_matrix_bridge} \newline
Abbildung \ref{conf_matrix_test} Verwirrungsmatrix für den Test Datensatz \dotfill \pageref{conf_matrix_test} \newline
Abbildung \ref{screenshot_testweb} Test des Webservice \dotfill \pageref{screenshot_testweb} \newline

\newpage