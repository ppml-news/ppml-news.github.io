# News in Privacy-Preserving Machine Learning

## February 2020

### Papers

- [Private Summation in the Multi-Message Shuffle Model](https://arxiv.org/abs/2002.00817)

## January 2020

### Papers

- [Approximating Activation Functions](https://arxiv.org/abs/2001.06370)

## February 2019

### Papers

- [Secure Evaluation of Quantized Neural Networks](https://eprint.iacr.org/2019/131)
- [TensorSCONE: A Secure TensorFlow Framework using Intel SGX](https://arxiv.org/abs/1902.04413)
- [Achieving GWAS with Homomorphic Encryption](https://arxiv.org/abs/1902.04303)
- [CodedPrivateML: A Fast and Privacy-Preserving Framework for Distributed Machine Learning](https://arxiv.org/abs/1902.00641)  
Interesting solution for offloading/out-sourcing model training to set of workers while ensuring strong privacy guarantees; based on Lagrange coded computations.
- [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046)

### Bonus

- [A Marauder's Map of Security and Privacy in Machine Learning](https://www.youtube.com/watch?v=5GQUXmSg8XU), a lecture on security and privacy. By [Nicolas Papernot](https://twitter.com/NicolasPapernot/status/1095149613344907264).
- [A Simple Explanation for the Existence of Adversarial Examples with Small Hamming Distance](https://arxiv.org/abs/1901.10861)  
Some of the greatest minds from cryptography join in on adversarial examples: "We develop a simple mathematical framework which enables us to think about this baffling phenomenon [and] explain why we should expect to find targeted adversarial examples in arbitrarily deep neural networks."


## January 2019

### Papers

- [Privacy-preserving semi-parallel logistic regression training with Fully Homomorphic Encryption](https://eprint.iacr.org/2019/101)
- [CaRENets: Compact and Resource-Efficient CNN for Homomorphic Inference on Encrypted Medical Images](https://arxiv.org/abs/1901.10074)  
Secure predictions using FHE with careful packing.
- [Differentially Private Markov Chain Monte Carlo](https://arxiv.org/abs/1901.10275)  
- [Improved Accounting for Differentially Private Learning](https://arxiv.org/abs/1901.09697)  
- [Secure Computation for Machine Learning With SPDZ](https://arxiv.org/abs/1901.00329)  
Looks at regression tasks using the general-purpose reference implementation and with active security.
- [Secure Two-Party Feature Selection](https://arxiv.org/abs/1901.00832)  
Privacy-preserving chi-squared test for binary feature selection from Paillier encryption.
- [Contamination Attacks and Mitigation in Multi-Party Machine Learning](https://arxiv.org/abs/1901.02402)  
Making models more robust to tainted training data by minimizing the ability to predict the providing parties.

### News

- [Videos from Hacking Deep Learning 2](https://www.youtube.com/watch?v=wPCr7iUIFKs&list=PL8Vt-7cSFnw1dO9kc2_SQQRchzpQQDYXy) online, including talks on adversarily attacks and privacy. [Via @BIUCrypto](https://twitter.com/BIUCrypto/status/1091245397283086336).
- [Videos from CCS'18](https://www.youtube.com/playlist?list=PLn0nrSd4xjjbyUeai0oevMrT8_IwnBo4R) online, including [presentation of ABY3](https://www.youtube.com/watch?v=X8l8XMNyHDM&list=PLn0nrSd4xjjbyUeai0oevMrT8_IwnBo4R). [Via @lzcarl](https://twitter.com/lzcarl/status/1090729920957837312).
- Simons Institute program on [Data Privacy: Foundations and Applications](https://twitter.com/ppml_news/status/1084778293428133889) kicked off this week with several workshops around differential privacy.
- [Program for SP'19](https://www.ieee-security.org/TC/SP2019/program-papers.html) is out with four accepted papers on differential privacy. [Via @IEEESSP](https://twitter.com/IEEESSP/status/1082680571833315330).

### Bonus

- [Deep Learning to Evaluate Secure RSA Implementations](https://eprint.iacr.org/2019/054)
- [Turbospeedz: Double Your Online SPDZ! Improving SPDZ using Function Dependent Preprocessing](https://eprint.iacr.org/2019/080)
- [Excellent summary](https://medium.com/dropoutlabs/privacy-preserving-machine-learning-2018-a-year-in-review-b6345a95ae0f) of what happened last year in the world of privacy-preserving machine learning by [Dropout Labs](https://twitter.com/dropoutlabsai/status/1083432828229038082).
- [Real World Crypto](https://rwc.iacr.org/2019/) happened this week, with (temporary?) [recordings](https://www.youtube.com/user/Regenegade) available on YouTube. Especially the talk on [Deploying MPC for Social Good](https://www.youtube.com/watch?v=W2thViwbEQQ&t=9496) has received significant attention, while the talk on [Foreshadow attack on Intel SGX](https://www.youtube.com/watch?v=4hq4yiVCopU&feature=youtu.be&t=2686) furthermore reminded us that enclaves are not perfect yet.

## 31 December 2018

### Papers

- [Fast Secure Comparison for Medium-Sized Integers and Its Application in Binarized Neural Networks](https://eprint.iacr.org/2018/1236)

- [Low Latency Privacy Preserving Inference](https://arxiv.org/abs/1812.10659)

### News

- Google AI team releases new [TensorFlow Privacy](https://github.com/tensorflow/privacy) library for training machine learning models with differential privacy for training data. [Via @NicolasPapernot](https://twitter.com/NicolasPapernot/status/1076195034209415173).

## 14 December 2018

### Papers

- [Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/abs/1812.02903)  
Update on concrete use of federated learning at Google; no secure computation nor differential privacy but including thoughts on dealing with unseen training data.

- [When Homomorphic Cryptosystem Meets Differential Privacy: Training Machine Learning Classifier with Privacy Protection](https://arxiv.org/abs/1812.02292)

- [Differentially Private User-based Collaborative Filtering Recommendation Based on K-means Clustering](https://arxiv.org/abs/1812.01782)

- [Privacy Partitioning: Protecting User Data During the Deep Learning Inference Phase](https://arxiv.org/abs/1812.02863)  
Optimising for privacy loss at early layers suggests pragmatic approach for protecting privacy of prediction inputs without cryptography nor DP.

- [A Review of Homomorphic Encryption Libraries for Secure Computation](https://arxiv.org/abs/1812.02428)

- [Private Polynomial Computation from Lagrange Encoding](https://arxiv.org/abs/1812.04142)

### News

- [NeurIPS workshop on Privacy Preserving Machine Learning](https://ppml-workshop.github.io/ppml/) happened this week with a very interesting [selection of papers](https://ppml-workshop.github.io/ppml/#papers).

- Intel's [HE Transformer for nGraph](https://github.com/NervanaSystems/he-transformer) released as open source!

### Bonus

- [Scaling Shared Model Governance via Model Splitting](https://arxiv.org/abs/1812.05979)


## 30 November 2018

### Papers

- [nGraph-HE: A Graph Compiler for Deep Learning on Homomorphically Encrypted Data](https://arxiv.org/abs/1810.10121)  
"One of the biggest accelerators in deep learning has been frameworks that allow users to describe networks and operations at a high level while hinding details ... A key challenge for building large-scale privacy-preserving ML systems using HE has been the lack of such a framework; as a result data scientists face the formidable task of becoming experts in deep learning, cryptography, and software engineering". Amen!

- [CHET: Compiler and Runtime for Homomorphic Evaluation of Tensor Programs](https://arxiv.org/abs/1810.00845)  
"In many respects, programming FHE applications today is akin to low-level assembly ... Our central hypothesis is that future applications will benefit from a compiler and runtime that targets a compact and well-reasoned interface". Amen! Also describes several ways on which the compiler can optimize encrypted computations.

- [Faster CryptoNets: Leveraging Sparsity for Real-World Encrypted Inference](https://arxiv.org/abs/1811.09953)  
Solid work on using weights quantization and other ML techniques to adapt neural networks for the encrypted setting, significantly improving performance relative to CryptoNets. Interestingly, second degree approximations of the Swish activation function are used over ReLUs and squaring. Gives plenty of references for those not coming from a ML background.

- [Privacy-Preserving Collaborative Preduction using Random Forests](https://arxiv.org/abs/1811.08695)  
Train models locally on independent data sets and apply ensemble techniques to serve private predictions using these.

- [FALCON: A Fourier Transform Based Approach for Fast and Secure Convolutional Neural Network Predictions](https://arxiv.org/abs/1811.08257 )  
Private predictions via FHE and GC. Interestingly, values are first convert to the frequency domain using the FFT and there’s a protocol for softmax.

- [The AlexNet Moment for Homomorphic Encryption: HCNN, the First Homomorphic CNN on Encrypted Data with GPUs](https://eprint.iacr.org/2018/1056)

- [A Fully Private Pipeline for Deep Learning on Electronic Health Records](https://arxiv.org/abs/1811.09951)

- [Distributed and Secure ML with Self-tallying Multi-party Aggregation](https://arxiv.org/abs/1811.10296)

### News

- List of accepted papers for [NeurIPS'18 privacy workshop](https://ppml-workshop.github.io/ppml/#papers) is out! [Via @mortendahlcs](https://twitter.com/mortendahlcs/status/1063053031246364673).

## 31 October 2018

### Papers

- [Privado: Practical and Secure DNN Inference](https://arxiv.org/abs/1810.00602)

## 28 September 2018

### Papers

- [Encrypted Databases for Differential Privacy](https://eprint.iacr.org/2018/860)

## 27 July 2018

### Papers

- [Efficient Logistic Regression on Large Encrypted Data](https://eprint.iacr.org/2018/662)

- [Round-Efficient Protocols for Secure Multiparty Fixed-Point Arithmetic](https://www.researchgate.net/publication/326652429_Round-Efficient_Protocols_for_Secure_Multiparty_Fixed-Point_Arithmetic)

## 27 June 2018

- [Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware](https://arxiv.org/abs/1806.03287)

- [TAPAS: Tricks to Accelerate (encrypted) Prediction As a Service](https://arxiv.org/abs/1806.03461)

- [DeepObfuscation: Securing the Structure of Convolutional Neural Networks via Knowledge Distillation](https://arxiv.org/abs/1806.10313)

- [ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models](https://arxiv.org/abs/1806.01246)

## 25 May 2018

### Papers

- [Logistic Regression over Encrypted Data from Fully Homomorphic Encryption](https://eprint.iacr.org/2018/462)

- [From Keys to Databases -- Real-World Applications of Secure Multi-Party Computation](https://eprint.iacr.org/2018/450)  
Jana: private SQL databases
Sharemind: secure analytics
Partisia, Sepior: auctions and key management
Unbound Technology: enterprise secrets

- [Minimising Communication in Honest-Majority MPC by Batchwise Multiplication Verification](https://eprint.iacr.org/2018/474)

- [SPDZ2k: Efficient MPC mod 2^k for Dishonest Majority](https://eprint.iacr.org/2018/482)  
To improve efficiency of MPC it is interesting to perform operations over rings that fit closely with native CPU instructions, as opposed to over e.g. a prime field. Doing so is straight forward when the attacker is honest-but-curious, and this paper addresses the case when he is fully malicious.

### News

- GDPR has come into effect!

- Slides from [UCL course on privacy enhancing technologies](https://www.benthamsgaze.org/2015/07/21/teaching-privacy-enhancing-technologies-at-ucl/) available. [Via @emilianoucl](https://twitter.com/emilianoucl/status/999537699764822017).

- [Keystone](https://keystone-enclave.github.io): An Open-source Secure Hardware Enclave. [Via @Daeinar](https://twitter.com/Daeinar/status/999457398598189056).

- The [reference SPDZ implementation](https://homes.esat.kuleuven.be/~nsmart/SCALE/) is being prepared for production. [Via @SmartCryptology](https://twitter.com/SmartCryptology/status/999951407938760704).

- Next week's [TPMPC workshop](http://www.multipartycomputation.com/tpmpc-2018) will be live-streamed if you happen to be elsewhere than Aarhus! [Via @claudiorlandi](https://twitter.com/claudiorlandi/status/998946571424235520).

### Bonus

- [Cautious Deep Learning](https://arxiv.org/abs/1805.09460)

## 18 May 2018

Small but good: we only dug up one paper this week but it comes with very interesting claims.

### Papers

- [SecureNN: Efficient and Private Neural Network Training](https://eprint.iacr.org/2018/442)  
Following recent approachs but reporting significant performance improvements via specialized protocols for the 3 and 4-server setting: the claimed cost of encrypted training is in some cases only 13-33 times that of training on cleartext data. Big factor in this is the avoidance of bit-decomposition and garbled circuits when computing comparisons and ReLUs.

## 11 May 2018

If anyone had any doubt that private machine learning is a growing area then this week might take care of that.

### Papers

Secure multiparty computation:

- [ABY3: A Mixed Protocol Framework for Machine Learning](https://eprint.iacr.org/2018/403)  
One of big guys in secure computation for ML is back with new protocols in the 3-server setting for training linear regression, logistic regression, and neural network models. Impressive performance improvements for both training and prediction.

- [EPIC: Efficient Private Image Classification (or: Learning from the Masters)](https://eprint.iacr.org/2017/1190)  
An update to work from last year on efficient private image classification using SPDZ and support vector machines. Includes great overview of recent related work.

Homomorphic encryption:

- [Unsupervised Machine Learning on Encrypted Data](https://eprint.iacr.org/2018/411)  
Implements K-means privately using fully homomorphic encryption and a bit-wise rational encoding, with suggestions for tweaking K-means to make it more practical for this setting. The TFHE library (see below) is used for experiments.

- [TFHE: Fast Fully Homomorphic Encryption over the Torus](https://eprint.iacr.org/2018/421)  
Proclaimed as the fastest FHE [library](https://tfhe.github.io/tfhe/) currently available, this paper is the extended version of previous  descriptions of the underlying scheme and optimizations.

- [Homomorphic Secret Sharing: Optimizations and Applications](https://eprint.iacr.org/2018/419)  
Further work on a hybrid scheme between homomorphic encryption and secret sharing: operations can be performed locally by each share holder as in the former, yet a final combination is needed in the end to recover the result as in the latter: "this enables a level of compactness and efficiency of reconstruction that is impossible to achieve via standard FHE".

Secure enclaves:

- [SecureCloud: Secure Big Data Processing in Untrusted Clouds](https://arxiv.org/abs/1805.01783)  
An joint European research project to develop a platform for pusing critical applications to untrusted cloud environments, using secure enclaves and supporting big data. Envisioned use cases from finance, health care, and smart grids.

- [SecureStreams: A Reactive Middleware Framework for Secure Data Stream Processing](https://arxiv.org/abs/1805.01752)  
Presents concrete work done in the above SecureCloud project, namely a high-level Lua-based framework for privately processing streams at scale using dataflow programming and secure enclaves.

Differential privacy:

- [Privately Learning High-Dimensional Distributions](https://arxiv.org/abs/1805.00216)  
Tackles the problem that privacy "comes almost for free when data is low-dimensional but comes at a steep price when data is high-dimensional" as measured in amount of samples needed. Two mechanisms are presented for learning respectively a multivariate Gaussian and a product distribution.

- [SynTF: Synthetic and Differentially Private Term Frequency Vectors for Privacy-Preserving Text Mining](https://arxiv.org/abs/1805.00904)  
A differentially private mechanism is used to prevent author re-identification in texts used for training models where anomymized feature vectors can be used instead of the actual body text. Concrete experiments include topic classification of newsgroups postings.

- [Distributed Differentially-Private Algorithms for Matrix and Tensor Factorization](https://arxiv.org/abs/1804.10299)  
Correlated noise is used to privately perform two common operations via a centralized but curious party or directly between data holders, respectively. Interestingly, the correlated noise is not uniform as in typical secure aggregation settings.

### Bonus

- [An Empirical Analysis of Anonymity in Zcash](https://arxiv.org/abs/1805.03180)
A little reminder that anonymity is hard.

## 27 April 2018

### Papers

- [Towards Dependable Deep Convolutional Neural Networks (CNNs) with Out-distribution Learning](https://arxiv.org/abs/1804.08794)  
"in this paper we propose to add an additional dustbin class containing natural out-distribution samples"
"We show that such an augmented CNN has a lower error rate in the presence of adversarial examples because it either correctly classifies adversarial samples or rejects them to a dustbin class."

- [Weak labeling for crowd learning](https://arxiv.org/abs/1804.10023)  
"weak labeling for crowd learning is proposed, where the annotators may provide more than a single label per instance to try not to miss the real label"

- [Decentralized learning with budgeted network load using Gaussian copulas and classifier ensembles](https://arxiv.org/abs/1804.10028)  
"In this article, we place ourselves in a context where the amount of transferred data must be anticipated but a limited portion of the local training sets can be shared.  We also suppose a minimalist topology where each node can only send information unidirectionally to a single central node which will aggregate models trained by the nodes"
"Using shared data on the central node, we then train a probabilistic model to aggregate the base classifiers in a second stage."

- [Securing Distributed Machine Learning in High Dimensions](https://arxiv.org/abs/1804.10140)  
Some results towards the issue of input pollution in federated learning, where a fraction of gradient providers may give arbitrarily malicious inputs to an aggregation protocol. "The core of our method is a robust gradient aggregator based on the iterative filtering algorithm for robust mean estimation".

## 20 April 2018

### Papers

- [Nothing Refreshes Like a RePSI: Reactive Private Set Intersection](https://eprint.iacr.org/2018/344)  
[PSI](https://www.youtube.com/watch?v=42pT3_Mqp7Q) was several applications in private data processing, including object linking in advertising and data augmentation. This paper takes a step towards mitigating exhaustive attacks where a party learns too much by simply asking for many intersections.

### News

- [Sharemind](https://sharemind.cyber.ee/), one of the biggest and earliest players pushing MPC to industry, has launched a [new privacy service](https://sharemind.cyber.ee/introducing-sharemind-hi/) based on [secure computation using secure enclaves](https://eprint.iacr.org/2016/1057) with the promise that it can handle big data. [Via @positium](https://twitter.com/positium/status/986178082812907520).

- Interesting [interview with Lea Kissner](https://gizmodo.com/meet-the-woman-who-leads-nightwatch-google-s-internal-1825227132), the head of Google's privacy team [NightWatch](https://www.buzzfeed.com/sheerafrenkel/google-has-a-secret-team-making-sure-its-products-are-safe?utm_term=.aw46WN654j#.dw461G6PqZ). Few details are given but <em>"She recently tried to obscure some data using cryptography, so that none of it would be visible to Google upon upload ... but it turned out that [it] would require more spare computing power than Google has"</em> sounds like techniques that could be related to MPC or HE. [Via @rosa.](https://twitter.com/rosa/status/986024500067106816)

- Google had two AI presentations at this year's RSA conference, one on fraud detection and one on adversarial techniques. [Via @goodfellow_ian](https://twitter.com/goodfellow_ian/status/987415311518392320).

### Bonus

- [Privacy-Preserving Multibiometric Authentication in Cloud with Untrusted Database Providers](https://eprint.iacr.org/2018/359)  
Relevant application of secure computation to authentication using sensitive data. Relative black box use of existing protocols yet experimental performance <1sec. 

- [Private Anonymous Data Access](https://eprint.iacr.org/2018/363)  
Interesting mix of [private information retrieval](https://en.wikipedia.org/wiki/Private_information_retrieval) and [oblivious RAM](https://en.wikipedia.org/wiki/Oblivious_ram): "We consider a scenario where a server holds a huge database that it wants to make accessible to a large group of clients while maintaining privacy and anonymity ... with the goal of getting the best of both worlds: allow many clients to privately and anonymously access the database as in PIR, while having an efficient server as in ORAM".

- [Adversarial Attacks Against Medical Deep Learning Systems](https://arxiv.org/abs/1804.05296)  
A discussion around some of the concrete consequences the medical profession may face from adversarial examples in machine learning systems with a warning of "caution in employing deep learning systems in clinical settings".

## 13 April 2018

### Papers

- [Differentially Private Confidence Intervals for Empirical Risk Minimization](https://arxiv.org/abs/1804.03794)  
Addresses the question of computing confidence intervals in a private manner, using either DP or [concentrated DP](https://arxiv.org/abs/1603.01887). Gives concrete examples and experiments using logistic regression and SVM.

### News

- Facebook host [privacy summit](https://research.fb.com/facebook-hosts-distinguished-faculty-for-privacy-summit/) but seem a bit sparse on details. [Via @sweis](https://twitter.com/sweis/status/984464406254829568).

### Bonus

- [PowerHammer: Exfiltrating Data from Air-Gapped Computers through Power Lines](https://arxiv.org/abs/1804.04014)  
More work on leaking data from air-gapped computers through obscure side-channels, this time through power lines by varying the CPU utilization, achieving bit rates of 10-1000 bit/sec for different attacks.

## 30 March 2018

### Papers

- [Private Nearest Neighbors Classification in Federated Databases](https://eprint.iacr.org/2018/289)  
Great read on custom MPC protocols allowing k-NN classification of a sample (such as document classification with cosine similarity) using a distributed data set, without leaking neither sample nor data set. This includes feature extraction, similarity computation, and top-k selection.

- [Chiron: Privacy-preserving Machine Learning as a Service](https://arxiv.org/abs/1803.05961)  
Interesting look at protecting both privacy of training data and model specifics via [secure enclaves](https://en.wikipedia.org/wiki/Software_Guard_Extensions). The technology is promising despite having experienced a few [issues recently](https://arxiv.org/abs/1802.09085) and e.g. avoids use of heavy cryptography.

- [Locally Private Bayesian Inference for Count Models](https://arxiv.org/abs/1803.08471)  
When applying differential privacy one may either ignore the fact that noise has been added to the data or try to take it into account; the latter is done here with good illustrations of the improvements this can give.

- [Hiding in the Crowd: A Massively Distributed Algorithm for Private Averaging with Malicious Adversaries](https://arxiv.org/abs/1803.09984)  
Interesting peer-to-peer protocol for privately computing the exact average of a distributed data set via gossiping directly between the peers. No heavy cryptography is used in case of honest peers, with a PHE-based extension for detecting malicious cheating.


- [Comparing Population Means under Local Differential Privacy](https://arxiv.org/abs/1803.09027)

- [Cloud-based MPC with Encrypted Data](https://arxiv.org/abs/1803.09891)  
Gives two schemes for private [*Model Predictive Control*](https://en.wikipedia.org/wiki/Model_predictive_control) by a central authority (who might have a better understanding of the environment than individual sensors), one based on PHE and another on MPC.


## 16 March 2018

### Papers

- [Model-Agnostic Private Learning via Stability](https://arxiv.org/abs/1803.05101)  
More work on ensuring privacy of training data via differential private query mechanisms. Compared to paper from a few weeks ago, this one focuses on "algorithms that are agnostic to the underlying learning problem [with] formal utility guarantees [and] provable accuracy guarantees".

- [Homomorphic Encryption for Speaker Recognition: Protection of Biometric Templates and Vendor Model Parameters](https://arxiv.org/abs/1803.03559)  
The Paillier cryptosystem is used to securely evaluate simplified similarity functions so users don't leak biometric information during authentication. Performance numbers included.

- [Efficient Determination of Equivalence for Encrypted Data](https://arxiv.org/abs/1803.03760)  
Reminder that even a simpler task such as privately linking identities and records together is relevant in industry.

### Bonus

- [The Morning Paper: When coding style survives compilation](https://blog.acolyer.org/2018/03/16/when-coding-style-survives-compilation-de-anonymizing-programmers-from-executable-binaries/)
Anonymity is hard! Random forests can be trained to identify your coding style from source code as well as compiled programs.

## 9 March 2018

### News

- The [2018 Gödel Prize](http://eatcs.org/index.php/component/content/article/1-news/2670-2018-godel-prize) is awarded to Oded Regev for his paper [On lattices, learning with errors, random linear codes, and cryptography](https://cims.nyu.edu/~regev/papers/qcrypto.pdf). This had a huge influence on later work in cryptography, not least homomorphic encryption. [Via @hoonoseme]( https://twitter.com/hoonoseme/status/971517058633601028).

- [OpenMined](https://openmined.org) is now maintaining a list of papers and tools around private machine learning: https://github.com/OpenMined/awesome-ai-privacy! [Via @iamtrask](https://twitter.com/iamtrask/status/971711677526892544).

- [Lab41](https://www.lab41.org/) has released a Python wrapper around Microsoft's [SEAL](http://sealcrypto.org) homomorphic encryption library: https://github.com/Lab41/PySEAL. [Via @mortendahl](https://twitter.com/mortendahlcs/status/971320764988346370
).

- The list of accepted contributed talks for this year's [Theory and Practice of MPC](http://www.multipartycomputation.com/tpmpc-2018) workshop has been announced. This is the definitive annual event dedicated to secure multi-party computation. [Via @claudiorlandi](https://twitter.com/claudiorlandi/status/970976361933365249
).


### Papers

- [Generating Differentially Private Datasets Using GANs](https://arxiv.org/abs/1803.03148)  
Interesting idea of using GANs to produce artificial differential privacy-preserving datasets from sensitive data that are safe to release for further training purposes. This is done on the client side, meaning there's no need for a trusted aggregator.

- [Faster Homomorphic Linear Transformations in HElib](https://ia.cr/2018/244)  
The mesters are at it again, giving algorithmic improvements to perhaps the most well-known homomorphic encryption library and thereby making it 30-75 times faster. 

- [Logistic Regression Model Training based on the Approximate Homomorphic Encryption](https://ia.cr/2018/254)  
Private fitting of several logisictic regression models on smaller genomic data sets using the [HEAAN](https://github.com/kimandrik/HEAAN) homomorphic encryption scheme. Approach is somewhat typical gradient descent and sigmoid polynomial approximation but with significant concrete performance improvements over other work using HEAAN.


### Blogs

- [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
Nothing to do with private machine learning, yet this is so neat that it warrents a mention. Go play!


## 2 March 2018

### News

- [@mvaria](https://twitter.com/mvaria)'s talk about a real-world application of MPC at this year's ENIGMA conference is online and well worth a watch: https://www.youtube.com/watch?v=d9rMokeYx9I. [Via @lcyqn](https://twitter.com/lcyqn/status/968638260774932480).


### Papers

- [Scalable Private Learning with PATE](https://arxiv.org/abs/1802.08908)  
Follow-up work to the [celebrated](https://blog.acolyer.org/2017/05/09/semi-supervised-knowledge-transfer-for-deep-learning-from-private-training-data/) Student-Teacher way of ensuring privacy of training data via differential privacy, now with better privacy bounds and hence less added noise. This is partially achieved by switching to Gaussian noise and more advanced (trusted) aggregation mechanisms.

- [Privacy-Preserving Logistic Regression Training](https://ia.cr/2018/233)  
Fitting a logistic model from homomorphically encrypted data using the Newton-Raphson iterative method, but with a fixed and approximated Hessian matrix. Performance is evaluated on the iDASH cancer detection scenario.

- [Privacy-Preserving Boosting with Random Linear Classifiers for Learning from User-Generated Data](https://arxiv.org/abs/1802.08288)  
Presents the *SecureBoost* framework for mixing boosting algorithms  with secure computation. The former uses randomly generated linear classifiers at the base and the latter comes in three variants: RLWE+GC, Paillier+GC, and SecretSharing+GC. Performance experiments on both the model itself and on the secure versions are provided.

- [Machine learning and genomics: precision medicine vs. patient privacy](https://arxiv.org/abs/1802.10568)  
Non-technical paper illustrating that secure computation techniques are finding their way into otherwise unrelated research areas, and hitting home-run with "data access restrictions are a burden for researchers, particularly junior researchers or small labs that do not have the clout to set up collaborations with major data curators".


### Blogs

- [Uber's differential privacy .. probably isn't](https://github.com/frankmcsherry/blog/blob/master/posts/2018-02-25.md)
[@frankmcsherry](https://twitter.com/frankmcsherry/status/968778164565626880) looks at Uber's [SQL differential privacy](https://github.com/uber/sql-differential-privacy) project and shares experience gained from implementing these things in Microsoft's [PINQ](https://www.microsoft.com/en-us/research/publication/privacy-integrated-queries/).


## 23 February 2018

### Papers

- [The Secret Sharer: Measuring Unintended Neural Network Memorization & Extracting Secrets](https://arxiv.org/abs/1802.08232)  
Concrete study of what a model can leak about sensitive information in the traning data. Perhaps not surprisingly, "only by developing and training a differential private model are we able to ... protect against the extraction of secrets".

- [Doing Real Work with FHE: The Case of Logistic Regression](https://ia.cr/2018/202)  
The heavyweights of homomorphic encryption apply [HElib](https://github.com/shaih/HElib) to logistic regression with a focus on implementing "optimized versions of many bread and butter FHE tools. These tools include binary arithmetic, comparisons, partial sorting, and low-precision approximation of complicated functions such as reciprocals and logarithms".

- [On the Connection between Differential Privacy and Adversarial Robustness in Machine Learning](https://arxiv.org/abs/1802.03471)
...

- [Reading in the Dark: Classifying Encrypted Digits with Functional Encryption](https://ia.cr/2018/206)  
Develops a functional encryption scheme for "efficient computation of quadratic polynomials on encrypted vectors" and applies this to private MNIST prediction (i.e. using a model trained on unencrypted data) via suitable quadractic models.

