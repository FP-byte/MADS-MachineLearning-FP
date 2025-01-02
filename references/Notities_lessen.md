1.1 Wat *tensors* zijn en voorbeelden voor de dimensies tm 5D
Tensor: vector of (2D) matrix van vectoren, (3D) (batch, hoogte, breedte), 4D (batch, hoogte, breedt, kanalen), 5D (batch, frames, hoogte, breedte, kanalen) bijv. van videostreaming
1.2  hoe neurale netwerken zijn opgebouwd (linear + activation), en hoe ze een extensie zijn van lineaire modellen (wanneer kun je ze stapelen)  
de functie y = f[x, params] is een opsommig van een line van de type Wx+b
1.3 neurale netwerken zijn een universal function
Een universele functiebenadering (universal function approximation) is een concept in de wiskunde en machine learning dat verwijst naar het vermogen van een model om elke continue functie te benaderen, mits het model voldoende complexiteit heeft.
Het idee is dat een model (zoals een neuraal netwerk) in staat is om elke willekeurige functie te benaderen tot een gewenste nauwkeurigheid, gegeven voldoende parameters, lagen, of tijd om te trainen.
Universal Approximation Theorem: een feedforward neuraal netwerk  met minstens één hidden layer (met voldoende aantal neuronen) kan een continue functie kan benaderen, op elke gesloten en begrensde interval, tot elke gewenste nauwkeurigheid. 
1.4 gradient descent: een optimalisatie-algoritme gebruikt in machine learning en deep learning om de parameters van een model (zoals de gewichten in een neuraal netwerk) aan te passen en zo de fout (of verlies) te minimaliseren. 

1.5 voordelen van een datastreamer voor grote datasets: data worden in batch aangeboden, om data in real-time of in kleine stukken (data chunks) te verwerken, in plaats van het laden van de volledige dataset in het geheugen.  Het biedt voordelen zoals efficiënt geheugengebruik, schaalbaarheid, real-time verwerking, en kostenbesparing, terwijl het ook in staat is om te werken met gedistribueerde gegevens en lage latentie biedt voor directe besluitvorming.
**Batch:** mechanisme om meer willekeur te krijgen. Elke iteratie de algoritme krijt een subset van de training data en berekent de gradient op basis van deze subset en niet de gehele dataset. Dit minibatch kan helpen om lokale minima te voorkomen en meer te richten naar global minima, wat meer generalisatie van de output functie betekent. De batch wordt gebruikt in de Stochastic gradient descent (SGD). Als de batch gelijk is aan de gehele dataset dan is de SGD gelijk aan de normale gradient descent. Een hele bewerking van de dataset is een epoch.
Voordelen SGD: voegt noise omdat de berekening op een kleine subset wordt gedaan, kan de lokale minima escapen
1.6 Wat de **invloed is van het aantal units in een dense layer** 
Meer units in een dense layer geven het model meer capaciteit om complexe relaties in de gegevens te leren. Elke unit in een laag kan als een soort "kenmerkdetector" worden gezien, dus meer units maken het model krachtiger in het herkennen van complexere patronen.
Als je te weinig units gebruikt, kan het model onderfitten (underfitting), wat betekent dat het model niet in staat is om de onderliggende patronen in de gegevens goed te leren.
Als je te veel units gebruikt, kan het model overfitten (overfitting), wat betekent dat het model te veel leert van de specifieke details in de trainingsgegevens en slecht generaliseert naar nieuwe, ongeziene gegevens.
Network capacity: aantal hidden units, meer hidden units meer benadering van complexe functies (width), hidden units zijn nodes
Network depth: aantal layers

Meer units leiden tot een groter aantal parameters in het netwerk, wat betekent dat er meer rekenkracht en geheugen nodig is voor zowel de training als de inferentie (voorspelling).
Dit verhoogt de trainingstijd, omdat het netwerk meer berekeningen moet uitvoeren tijdens elke stap van de gradient descent-optimalisatie.
De benodigde tijd voor backpropagation en gewicht-updates neemt toe naarmate het aantal units in een laag toeneemt.
Wanneer we modellen trainen, zoeken we naar de parameters die de best mogelijke mapping/koppeling van input naar output voor de betreffende taak produceren. De loss function geeft een mate van mismatch tussen voorspelling en output. We zoeken parameters die dit veschil zo klein mogelijk maken zodat de input gemapt is aan de output met zo min mogelijk verschil.
Parameters kunnen zijn: intercept of slope, d.w.z. de slope waarden == gewichten

1.7 **Hoe een gin file werkt** en wat de voordelen daarvan zijn: in een gin file worden de hyperparameters van een neurale netwerk mee gegeven bij het aanmaken van een netwerk. Daar kunnen de kenmerk ook aangepast worden voor hypertuning. 
1.8 Waarom we **train-test-valid splits** maken
Trainingsset: Wordt gebruikt om het model te trainen.
Validatieset: Wordt gebruikt om het model te evalueren en hyperparameters af te stemmen.
Testset: Wordt pas gebruikt na het trainen en afstemmen van het model om de uiteindelijke prestaties te testen.
Een gebruikelijke verdeling is bijvoorbeeld:
70% voor training
15% voor validatie
15% voor testen
Hyperparameters zijn instellingen van het model die niet tijdens het trainen worden geleerd, zoals de leersnelheid, het aantal lagen in een neuraal netwerk, het aantal eenheden in elke laag, de batchgrootte, en meer.
Je kunt de validatieset gebruiken om verschillende instellingen van hyperparameters uit te proberen en te testen om te zien welke het beste presteren. Dit proces wordt vaak hyperparameteroptimalisatie genoemd.
Een veelgebruikte aanpak om hyperparameters af te stemmen is grid search of random search, waarbij je verschillende combinaties van hyperparameters test en de configuratie kiest die de beste resultaten oplevert op de validatieset.
**Validatie**: Nadat het model is getraind, wordt de nauwkeurigheid geëvalueerd op de validatieset.
Dit helpt ons te begrijpen hoe goed het model presteert met de gegeven hyperparameters.
**Hyperparameteroptimalisatie**: Bijv. we testen verschillende leersnelheden (learning_rate_init) om te zien welke het beste werkt op de validatieset. Het model met de hoogste nauwkeurigheid op de validatieset wordt beschouwd als het beste model.
**Testen**: Het beste model wordt uiteindelijk geëvalueerd op de testset. De testset wordt pas gebruikt nadat de hyperparameters zijn geoptimaliseerd en het model is getraind op de trainingsset.

1.9 Wat een Activation function is, en kent er een aantal (ReLU, Leaky ReLU, ELU, GELU, Sigmoid) 
Een **activation function** (activeringsfunctie) is een wiskundige functie die de output van een neuron in een neuraal netwerk bepaalt. Het zorgt ervoor dat het netwerk niet-lineair (non-linear) wordt, waardoor het in staat is om complexe patronen en relaties in de data te leren.
*ReLU* is de meest gebruikte activatiefunctie in diepe netwerken. Het geeft de input door als de waarde groter is dan 0, en anders geeft het 0 terug. Het is eenvoudig en leidt tot snellere training, maar kan het probleem van "dode neuronen" veroorzaken, waarbij neuronen nooit geactiveerd worden als ze negatieve inputs ontvangen. 
*Leaky ReLU* is een variant van ReLU die voorkomt dat neuronen "doodgaan" door negatieve waarden een kleine, niet-nul output te geven (in plaats van 0). Dit zorgt ervoor dat de neuron ook voor negatieve inputs actief blijft.
*ELU* is een activatiefunctie die voor positieve waarden werkt zoals ReLU, maar voor negatieve waarden wordt de output een exponentiële functie die de negatieve output verzacht. Het helpt bij het verminderen van de bias van de activatiefunctie en het versnellen van de convergentie in vergelijking met ReLU.
*GELU* is een activatiefunctie die een combinatie is van een sigmoïde en een ReLU, en het heeft een gladder verloop dan ReLU. Het is gebaseerd op een probabilistische benadering en wordt vaak gebruikt in Transformer-netwerken zoals BERT. GELU is wiskundig geavanceerder dan ReLU, maar werkt vaak beter in moderne netwerken.
De **Sigmoidfunctie** is een S-vormige (sigmoïde) curve die de output beperkt tot het interval tussen 0 en 1. Het wordt vaak gebruikt in de outputlaag van classificatiemodellen voor binaire classificatie, omdat de output kan worden geïnterpreteerd als de waarschijnlijkheid van een klasse.
ReLU: Eenvoudig, snel, maar kan leiden tot "dode" neuronen.
Leaky ReLU: Voorkomt dode neuronen door een kleine negatieve helling te geven.
ELU: Een exponentiële variant die negatieve inputs verzacht, waardoor sneller convergeren mogelijk is.
GELU: Een gladde, probabilistische activatiefunctie die goed werkt in moderne netwerken zoals Transformers.
Sigmoid: Beperkt de output tussen 0 en 1, vaak gebruikt voor binaire classificatie.
1.10 wat een loss functie is:
Een loss functie (of kostenfunctie) berekent de fout of het verschil meet tussen de voorspelde waarden van een model en de werkelijke waarden (targets) in een dataset
Mean Squared Error (MSE):
Toepassing: Veel gebruikt voor regressieproblemen.
**MSE** berekent het gemiddelde van de kwadraten van de verschillen tussen de werkelijke en voorspelde waarden. Dit betekent dat grotere fouten zwaarder worden gestraft, wat kan helpen om grote afwijkingen te vermijden.
**Cross-Entropy Loss**(ook bekend als Log Loss): 
Deze loss functie meet de afwijking tussen de werkelijke klasse en de voorspelde waarschijnlijkheid. 
Toepassing: Veel gebruikt voor classificatieproblemen, vooral binaire en multi-class classificatie.
Afhankelijk van het type probleem (bijvoorbeeld classificatie of regressie) kies je een geschikte loss functie zoals Mean Squared Error (MSE) voor regressie of Cross-Entropy Loss voor classificatie.
1.11 Hoe deep learning past binnen de geschiedenis van AI, en wat deep learning kenmerkt ten opzichte van de rest.
First neuron idea is from 1943, perceptron (1958), 1980's backpropagatio developed which started a new development  
1.12 **stappen van het trainen van een NN*-: datapreparatie, trainbare gewichten, predict, lossfunctie, optimizers 
A supervised learning model is a fuction y = f[x, params] that relates input x to input y by parameters. To train the model we define a loss function over a training dataset which quantifies the mismatch between the prediction of f and the observed outputs y. We search the parameters that minimize the loss. We evaluate on a different set of data (test) to see how well it generalizes. 

2.1 - Verschillende loss functies (MSE en RMSE, Negative Log Likelihood, Cross Entropy Loss, Binary Cross Entropy loss) en in welke gevallen je ze moet gebruiken of vermijden:

**RMSE**
### MSE (for regression models):
Mean Squard Error, neemt de som van alle verschillen n tussen y en predicted y tot de macht 2 (alleen positief), en gedeeld door het aantal data n. Het is default voor regressie loss:

$$MSE = \frac{1}{n}\sum_{i=1}^n (Y_i - \hat{Y}_i)^2$$

This is the mean $\frac{1}{n}\sum_{i=1}^n$ of the squared error $(Y_i - \hat{Y}_i)^2$ 
But torch has already implemented that for us in an optimized way:
````
loss = torch.nn.MSELoss()
loss(yhat, y)
````
### Negative log likelihood. 
**Likelihood**: In probabilistische modellen beschrijft likelihood hoe waarschijnlijk het is dat een bepaald set van parameters de geobserveerde data heeft gegenereerd. Stel je voor dat je een model hebt dat een kansverdeling voorspelt voor een bepaalde dataset. De likelihood is de kans dat je de werkelijke observaties ziet, gegeven de parameters van je model.
De **Negative Log Likelihood** is simpelweg de negatieve waarde van de logaritme van de likelihood.
De reden dat we de log gebruiken, is om te zorgen dat de berekeningen eenvoudiger zijn (de kans op meerdere gebeurtenissen wordt vermenigvuldigd, maar de log van een product is de som van de logs, wat rekentechnisch handiger is).
Het "negatieve" teken wordt toegevoegd omdat we vaak een verliesfunctie willen minimaliseren tijdens de training van het model. Log-likelihood is echter een maat voor de waarschijnlijkheid (die we willen maximaliseren), dus door het negatief te maken, veranderen we het in een verliesfunctie die we willen minimaliseren.
The function is:

$$NLL = - log(\hat{y}[c])$$

Or: take the probabilities $\hat{y}$, and pick the probability of the correct class $c$ from the list of probabilities with $\hat{y}[c]$. Now take the log of that.

The log has the effect that predicting closer to 0 if it should have been 1 is punished extra.
````
loss = torch.nn.NLLLoss()
loss(yhat, y)
````
Het is te bewijzen dat negative log-likehood en cross entropy equivalent zijn.
**Cross Entropy Loss**: 
Cross-entropy meet de "afstand" tussen twee waarschijnlijkheidsdistributies:
De echte distributie (meestal de werkelijke labels, vaak één-hot gecodeerd).
De voorspelde distributie (de waarschijnlijkheden die het model toekent aan de verschillende klassen).
Het meet het verschil tussen twee waarschijnlijkheidsdistributies: de voorspelde waarschijnlijkheden en de werkelijke waarschijnlijkheden (de echte labels). In eenvoudige termen berekent de Cross Entropy Loss  hoe goed de voorspelde klassen waarschijnlijkheden overeenkomen met de werkelijke klassenlabels == vind het minimum verschil in waarchijnlijkheid.
Als de voorspelde waarschijnlijkheid  dicht bij 1 ligt voor de juiste klasse (en dicht bij 0 voor de andere klassen), zal het verlies klein zijn (omdat og(1)=0).
Als de voorspelde waarschijnlijkheid ver van 1 ligt (d.w.z. onjuiste voorspellingen), wordt het verlies groter, wat het model aanmoedigt om zich aan te passen in de volgende iteraties.
Categorical CEL: output is een verdeling van probabiliteit per categorie. LogSoftmax is dan ook geintegreerd.

Binary Cross Entropy loss: output is een waarde tussen 0 en 1 (wel of niet de categorie)
**momenutum**: updates the parameers with a weighted combination of the gradient computed from the current batch en the direction moved in the prevous step. The weights of the past gradient gets smaller.
**adam**: normalize the gradients so we move a fixed distance governed by the learning rate in each direction. This to avoid that big gradients gets big changes en small gradients get small changes (because of the calculation without normalization).
**L2 regularization**: applied to weights but not biases, weight decay term. The effect is to encourage smaller weights so the output function is smoother and less overfitted. Small weight mean a small variance and therefore the error reduces because the network do not overfit the data. When the netwok is overparametirized the regularization term will favour functions that interpolate smoothly between nerby points.
2.2 - Dimensionality probleem is bij afbeeldingen van Dense Neural Networks: fully connected networks hebben het nadeel dat ze niet goed werken voor grote inputs zoals beelden waarin de neuronen samen specifieke waarnemingen moeten nemen, er is geen nut dat elke neuroon meedoet met elke onderdeel. Eerder is dat handiger om een beeld in parallel te analiseren en steeds in grotere gebieden te integreren. (p. 51). Verder shallow dense networks zijn trager in het trainen en kunnen minder goed generaliseren.
2.3 - Hoe Convolutions dat dimensionality probleem oplossen: beelden hebben een hoge dimensie (bij. 224x224 maken in tot 150.528 input dimensies). Hidden layers in fully connected networks zijn groter dan de input, dit kan betekenen meer dan 22 miljard neurone.
Pixels van beelden die dichtbij zijn,zijn ook statistisch aan elkaar verbonden. FC moeten patronen leren per pixel en in elke mogelijke positie terwijl bij beelden spraken is van samengestelde patronen in verschillende posities. CNN leren door gebruik te maken van minder parameters (gewichtjes) en kunnen omgaan met geaugmenteerde beelden (rotaties, uitgestrekt, gespiegeld enz.) en kunnen hierdoor patronen beter abstraheren.
**Filter (convolutional kernel)**: convolutional layer met dezelfde gewichten op elke positie, berekent een gewogen som van de dichtbij inputs.
Kernel size kan verhoogt worden maar blijf een oneven aantal zodat gecentreerd is op de huidige pixel positie. Grotere kernel== meer gewichten. Via dilatation rate kunnen 0 tussen de gewichten toegevoegd worden zodat er een grotere kernel gebuikt wordt maar niet meer gewichten.
stride: (stap), stride = 1, elke positie word door de kernel berekend (groote blijft gelijk), stride =2 berekende de helft van de pixels
convolutional layer: berekent de output door de input te bewerken en de bias toe te voegen en elke resultaat via een activatiefunctie door te geven. Elke convolutie genereert a nieuwe set van hidden variable (feature map of channel).
Example: Kernel and Convolution in Action
Let’s say we have a 3x3 kernel and a 5x5 input image:

Input Image (5x5):

[1 2 3 0 1
 4 5 6 1 2
 7 8 9 2 3
 1 3 5 1 4
 2 4 6 0 1]


Kernel (3x3):

[ 1 0 −1
  1 0 −1
  1 0 −1]
​
Now, let's apply the convolution operation with a stride of 1 and no padding. Starting from the top-left corner, we calculate the dot product between the kernel and the input image at that position. The first dot product is:

(1∗1)+(0∗2)+(−1∗3)+(1∗4)+(0∗5)+(−1∗6)+(1∗7)+(0∗8)+(−1∗9)=−6
This value (-6) will be placed in the output feature map. The kernel then moves to the next position, sliding over by one pixel (because of stride 1), and the process is repeated until the entire image is processed.

De kernel in een CNN is een kleine matrix die over de invoergegevens beweegt en convolutie-operaties uitvoert om belangrijke kenmerken te extraheren. De kernel is verantwoordelijk voor het detecteren van specifieke patronen in de gegevens, en meerdere kernels worden meestal gebruikt om verschillende kenmerken op verschillende abstractieniveaus in een netwerk te detecteren. Deze geleerde patronen of kenmerken worden vervolgens doorgegeven aan diepere lagen voor verdere verwerking en classificatie.

2.4 - Wat maxpooling doet: vermindert de dimensies door een gemiddelde of een maximum over een pool size (window) te berekenen.
Max pooling vermindert de ruimtelijke dimensies van de feature map, terwijl de belangrijkste kenmerken behouden blijven door de maximumwaarde uit een venster van waarden te nemen.
Het helpt de grootte van het netwerk te verkleinen, verhoogt de rekenkundige efficiëntie en verbetert de translatie-invariantie.
Stride bepaalt hoeveel de pooling-window beweegt bij elke stap.
De output feature map na pooling heeft kleinere ruimtelijke dimensies, maar **behoudt de diepte (aantal kanalen)**.

2.5 - Wat een stride en padding zijn  
2.6 - Wat een Convolution van 1x1 filter doet: Het 1x1 filter wordt toegepast op elke pixel van de input feature map, om de channels te verandere. Berekend een gewogen som van alle channels op een pixel positie. Imdat de filtermaat 1x1 is, kijkt het alleen naar de waarden van één pixel per keer en hierdoor is het mogelijk om de diepte (aantal kanalen) van de feature map te veranderen. 
Lineaire transformatie: De 1x1 filter voert een lineaire transformatie uit door een gewichten matrix toe te passen op de verschillende kanalen (diepte) van de input. Hierdoor kunnen nieuwe, gecombineerde representaties van de input worden gemaakt zonder de ruimtelijke dimensies te beïnvloeden. 
2.7 - Kent ruwweg de innovaties die de verschillende architecturen doen: AlexNet, VGG, GoogleNet, ResNet, SENet 
Alexnet: eerste CNN die goed werkte (op Imagenet dataset). Vorzien van 8 hidden layers, Relu, eerste 5 convoluitonals en rest FC.
VGG: dieper dan Alexnet, 18 lagen
GoogleNet: hier werd de 1x1 kernel bedacht, nog diepere netwerk. Dit toonde wel aan dat meer lagen niet altijd beter performance opleverden. Dit heeft te maken met weight decay, die is groter naar mate meer lagen er zijn. Een kleine weiziging in de beginlagen kan grote gevolgen hebben op diepere lagen. (shattered gradients)
ResNet: residual or skip connection zijn aftakkingen van een berekening waar de input van elke layer is weer toegevoegd aan de output op recursieve wijzen. De residual connection generre een ensambel van kleinere netwerken en hun output is samengevoegd (opgeteld) in een resultaat. 
2.8 - Begrijpt wat overfitting is en hoe regularisatie (batchnorm, split van training/test/validation, dropout, learning rate) helpt.  

De student kan:  
2.9 - reproduceren hoe een Convolution schaalt (O(n)) ten opzicht van een NN (O(n^2))
De schaling van een Convolutional Neural Network (CNN) in vergelijking met een Fully Connected Neural Network (FCNN) kan worden begrepen door naar het aantal berekeningen of parameters te kijken die nodig zijn om een input te verwerken. 
1. Fully Connected Neural Network (FCNN) Schaling
In een fully connected neural network (FCNN), ook wel een dense network, zijn alle neuronen in elke laag verbonden met alle neuronen in de volgende laag. Dit betekent dat het aantal verbindingen (parameters/gewichten) snel toeneemt naarmate je het aantal neuronen vergroot.

Stel dat je een inputvector hebt van lengte n en dat je een verborgen laag hebt met m neuronen. De matrixvermenigvuldiging tussen de inputvector en de gewichtenmatrix van deze laag heeft O(n * m) berekeningen nodig. Als je meerdere lagen hebt, wordt de tijdcomplexiteit van het hele netwerk O(n^2) als het aantal lagen in het netwerk lineair toeneemt.

Voorbeeld:

Stel je voor dat je een netwerk hebt met 1000 inputneuronen en 1000 neuronen in de eerste verborgen laag. De aantal parameters (gewichten) is dan 1000 * 1000 = 1.000.000.
Als je een netwerk hebt met 10 lagen, neemt het aantal berekeningen exponentieel toe, omdat elke laag met elke andere laag volledig verbonden is. De tijdcomplexiteit kan O(n^2) worden, afhankelijk van het aantal neuronen in elke laag.
2. Convolutional Neural Network (CNN) Schaling
In een Convolutional Neural Network (CNN), daarentegen, worden de berekeningen op een andere manier uitgevoerd. CNN's gebruiken convoluties om lokale patronen in de input te detecteren, wat resulteert in een betere schaling dan in een volledig verbonden netwerk. Dit komt omdat een convolutioneel filter maar een klein gedeelte van de input hoeft te verwerken (in plaats van de volledige input), en dit filter kan herhaaldelijk over de input worden toegepast om kenmerken te extraheren.

De schaling van een CNN wordt meestal beschreven als O(n), omdat de convolutionele filters de input lokaal verwerken en geen volledige matrixvermenigvuldiging nodig hebben zoals in een FCNN.

Waarom is het O(n) voor een CNN?
Convolutie (filter toepassen): Stel dat je een 2D afbeelding van grootte n x n hebt (bijvoorbeeld 28x28 pixels), en je past een klein filter toe, bijvoorbeeld een 3x3 filter. In plaats van dat elke pixel met elke andere pixel wordt verbonden (zoals bij een volledig verbonden netwerk), wordt het filter slechts op een lokaal gebied toegepast (bijvoorbeeld 3x3 van de pixels) en schuift het filter door de afbeelding.

Bij elke toepassing van het filter op een 3x3 regio, worden er slechts 9 berekeningen uitgevoerd.
Het filter beweegt over de afbeelding met een bepaalde stapgrootte (stride), en daarom is het aantal convoluties (en dus de berekeningen) lineair gerelateerd aan de inputgrootte O(n).  
Gewichten (Weights): Dit zijn specifieke parameters die de verbindingen tussen neuronen in een neuraal netwerk vertegenwoordigen.
Parameters: Dit is de algemene term die alle waarden omvat die het netwerk leert, inclusief gewichten én biases.

2.10 - Een configureerbaar Convolution NN bouwen en handmatig hypertunen  
2.11 - MLFlow gebruiken naast gin-config  
2.12 - een Machine Learning probleem classificeren als classificatie / regressie / reinforcement / (semi)unsupervised  
2.13 - begrijpt welke dimensionaliteit een convolution nodig heeft en wat de strategieen zijn om dit te laten aansluiten op andere tensors  

**Adjusting the Number of Filters**
Gradually increasing filters: It's common practice to start with a small number of filters in the first layer and progressively increase them in subsequent layers. This is because the lower layers typically capture simpler, more general features (e.g., edges, colors), while the higher layers capture more complex, abstract features.
In the context of Convolutional Neural Networks (CNNs), filters (also known as kernels) are small matrices or weights that are used to scan (or convolve) over the input data in order to detect specific features. These features can be patterns like edges, textures, or shapes in the input, and they are learned during the training process.
