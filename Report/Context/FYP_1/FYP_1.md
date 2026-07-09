<table>
<tbody>
<tr class="odd">
<td><blockquote>
<p><strong>DE-42 (MTS) HASSAN, NUMAN, FIZA, IFRAH</strong></p>
</blockquote></td>
<td><p><span class="underline">­­­­­­­­­­­­­­­­­­</span></p>
<blockquote>
<p><strong>HARP HUMANOID ASSISTANT ROBOTIC</strong></p>
<p><strong>PLATFORM</strong></p>
</blockquote>
<p><img src="FYP_1/media/image1.emf" style="width:1.70417in;height:1.66667in" /></p>
<p><strong>COLLEGE OF</strong></p>
<p><strong>ELECTRICAL AND MECHANICAL ENGINEERING NATIONAL UNIVERSITY OF SCIENCES AND TECHNOLOGY RAWALPINDI</strong></p>
<p><strong>2024</strong></p></td>
</tr>
</tbody>
</table>

![](FYP_1/media/image2.jpeg)

> **DE-42 MTS**
> 
> **PROJECT REPORT**
> 
> **<span class="underline">HARP HUMANOID ASSISTANT ROBOTIC</span>**
> 
> **<span class="underline">PLATFORM</span>**
> 
> Submitted to the Department of Mechatronics Engineering
> 
> in partial fulfillment of the requirements
> 
> for the degree of
> 
> **Bachelor of Engineering**
> 
> **in**
> 
> **Mechatronics**
> 
> **2024**
> 
> **Sponsoring DS: Submitted By:**

Dr. Anas Bin Aqeel Hassan Rizwan

Asst. Prof. Kanwal Naveed Numan Siddique

> Lec. Usman Asad Ifrah Sajjad

Fiza Hassan

**<span class="underline">ACKNOWLEDGMENTS</span>**

To conduct this Thesis on humanoid robot, we would like to express our gratitude to Allah Almighty, who bestowed His blessings to carry out this extensive research, designing and completion of our project. Further, we would like to extend our humble gratitude to our supervisor, Dr. Anas Bin Aqeel, whose unwavering support, invaluable guidance, and insightful feedback have been instrumental in shaping this thesis. His expertise and encouragement have inspired us to strive for excellence in our research endeavors.

We also extend our heartfelt appreciation to our co-supervisors, Asst. Prof. Kanwal Naveed and Lec. Usman Asad, for their dedicated mentorship and constructive criticism. Their expertise and encouragement have enriched our understanding and enhanced the quality of our work.

Lastly, we are grateful to NUST H-12 for giving us a sponsorship funded flagship project and we would like to acknowledge the support and encouragement of our parents and friends. Their unwavering belief in our abilities, endless encouragement, and unconditional support have been a constant source of strength throughout the arduous process of undertaking this final year project. Their words of wisdom and encouragement have been a guiding light, motivating us to overcome challenges and strive for excellence.

**<span class="underline">ABSTRACT</span>**

> As the world is leaning towards autonomous devices and robots, the influence of technology is inevitable. There’s only one thing that’s constant in this world, and that is change. Consider a humanoid robot which assists you in any industry, especially healthcare, where it’s made to be more engaging and with better interactive skills through an interactive application. Despite having developed frameworks being successful for the short-term, a multifunctional long-term influence is seen through our product. The robot's capabilities encompass various aspects of human-robot interaction, including navigation, communication, and perception. By using Gemini along with specific dataset fed, the chatbot responds accordingly. Placing depth camera for visual input, the trained model detects the sentiments of the interacting user and imitates them by displaying a similar emotion on the lcd screen placed on its head. Our work contributes to the field of robotic assistive technology by demonstrating the feasibility and effectiveness of integrating humanoid features onto a mobile platform for receptionist applications.

# **<span class="underline">Table of Contents</span>**

ACKNOWLEDGMENTS i

ABSTRACT ii

TABLE OF CONTENTS iii

LIST OF FIGURES viii

LIST OF TABLES xii

LIST OF SYMBOLS xiii

Chapter 1 – INTRODUCTION 1

1.1 Overview 1

Chapter 2 – LITERATURE REVIEW 2

2.1 DEVI: Open-source Human-Robot Interface for Interactive Receptionist System 2

2.1.1 System Architecture and Design 2

2.1.1.1 Hardware Layer 2

2.1.1.2 Robot Intelligence Core 2

2.1.2 Experiments and Results 3

2.2 Talking Receptionist Robot 5

2.2.1 Methodology 5

2.2.2 Experiments and Results 6

2.3 Implementation of Voice Based Home Automation System Using Raspberry Pi 6

2.3.1 System Design and Architecture 6

2.3.1.1 Proposed System 7

2.3.1.2 Proposed System Workflow 7

2.3.1.3 Methodology 8

2.3.2 Experiment and Results 9

2.4 Design Modeling and Fabrication of Human-Humanoid Robot Communication 9

2.4.1 System Architecture and Design 9

2.4.1.1 Methodology 9

2.4.2 Results and Discussion 10

Chapter 3 – HARDWARE METHODOLOGY 11

3.1 Introduction 11

3.1.1 Overview of design process 11

3.2 Design conceptualization 11

3.2.1 Initial conceptual phase 11

3.2.2 Methodology to generate design concepts 12

3.3 SolidWorks modelling 13

3.3.1 People Bot SolidWorks model 13

3.3.2 Concept Exploration 14

3.3.3 Iterative design process 16

3.4 Parts 17

3.4.1 Final Torso Assembly 20

3.5 Design and implementation of head 21

3.5.1 Design Process 21

3.5.1.1 Initial Concept and Requirements 21

3.5.1.2 Functional Integration 21

3.5.2 SolidWorks Design 21

3.5.3 Manufacturing of Head 23

3.5.4 Final Assembly 24

3.6 Design Challenges and Strategies 25

3.7 Design Validation 25

3.7.1 Finite Elements Analysis 25

3.7.2 Static Structural Analysis 25

3.7.3 ANSYS Workbench 25

3.8 Static Analysis Using ANSYS 26

3.8.0.1 Simulation Setup 26

3.8.0.2 Analysis Process 26

3.8.1 Results 29

3.8.2 Including the torso. 31

3.8.3 Mesh Convergence and Importance 31

3.8.3.1 Initial Setup 31

3.8.3.2 Initial Meshing 31

3.8.3.3 Run Initial Analysis 32

3.8.3.4 Mesh Refinement 32

3.8.3.5 Re-Run the Analysis 33

3.8.3.6 Convergence Test 34

3.8.3.7 More Sophistication if Needed. 34

3.8.3.8 Final Validation 35

3.9 Results and Analysis 36

3.10 Dynamic Analysis 38

3.10.1 Final Validation 39

3.11 Manufacturing 40

3.11.1 Available materials 40

3.11.2 Available Methods 42

3.11.3 Selected Methods 45

3.11.3.1 Laser Cutting and Welding 45

3.11.3.2 3D Printing 46

3.12 Assembly 48

3.12.1 Initial assembly 48

3.12.2 Final Torso Assembly 49

3.12.3Assembly Challenges and Solutions 51

Chapter 4 - SYSTEM DESIGN AND IMPLEMENTATION 53

4.1 System design and implementation 53

4.1.0 Dataset Preparation 53

4.1.1 Emotion Recognition Mode 53

4.1.2 Training 53

4.1.3 Model of Gender Recognition 54

4.1.4 Real-Time Emotion and Gender Detection 56

4.1.5 Displaying Emoticons on LCD 56

4.1.6 Emoticons/Avatars 56

4.1.7 Raspberry Pi 4B 60

4.1.8 Functional overview 61

4.1.9 Operation 62

4.1.10 Libraries and Tools 63

4.1.11 Technical Details of CNN 64

4.1.11.1 Convolutional Neural Networks (CNNs) 64

4.1.11.2 Pooling Layers 64

4.1.11.3 Flattening and Fully Connected Layers 64

4.2 speech synthesis 64

4.2.1 Need for Natural Human-Computer Interaction 65

4.2.2 Role of Speech in communication 65

4.2.3 Speech Synthesis and Recognition 65

4.2.3.1 techniques used in Speech Synthesis and Recognition 65

4.2.3.2 Methodology 65

4.2.3.3 Overview of Speech Synthesis and Recognition 65

4.2.3.4 Implementation details 65

4.2.3.5 Accuracy and Error Handling 67

4.2.4 Python chatbot Development: A simple Starting Point 68

4.2.5 Capabilities and Limitations of Gemini and Open AI 69

4.2.6 Implementation of Gemini Chatbot 70

4.2.7 Introduction to RAG-Retrieval Augmented Generations 72

4.2.8 RAG Architecture 73

4.2.9 Integrating RAG and Gemini Chatbot 73

4.2.9.1 Design Considerations for Combining Systems 73

4.2.9.2 Implementation Details of the Integrated Chatbot 75

4.2.10 Results and Evaluation 77

4.2.10.1 Evaluation Metrics 77

4.2.10.2 Comparison of Response Quality and Fluency 77

4.2.10.3 Analysis of Efficiency and Resource Usage 78

4.2.10.4 Effectiveness of Combining RAG with Large Language Models 78

4.2.10.5 Interactive Chatbot Performance 79

4.2.10.6 Effectiveness in Handling Complex Queries 79

4.2.11 Limitations and Future Work 79

4.2.11.1 Challenges Encountered During Development 79

4.2.11.2 Potential Areas for Improvement and Further Research 80

4.3 User Graphical Interface 80

4.3.1 Web Application Technologies for Users 80

4.3.1.1 HTML Structure 81

4.3.1.2 Contact Us 81

4.3.1.3 Direction and Availability 81

4.3.1.4 Appointments 81

4.3.2 CSS Styling 81

4.3.3 JavaScript Interactivity 81

4.3.4 Speech Recognition 81

4.3.5 API Endpoints 81

4.3.6 Integration LangChain and Google Generative AI 82

4.3.7 Implementation Details 82

4.3.7.1 Frontend (HTML, CSS, JavaScript) 82

4.3.7.2 Backend (Flask, Py-Term, OpenAI Text Generation, Google Generative AI) 83

4.3.7.3 Code Integration and Workflow 83

4.3.8 Process Overview 83

4.3.8.1 Form Submission 83

4.3.8.2 Data Extraction 84

4.3.9 Conclusion 84

4.3.10 Code Snippets 84

4.3.10.1 Code A 84

4.3.10.2 Code B 88

4.3.10.3 Code C 90

Chapter 5 - CONCLUSION AND FUTURE WORK 92

5.1 Significance of Humanoid Assistive Robotic Plateform 92

5.2 Conclusion 92

5.3 Future Work 92

REFRENCES……………………………………………………………………………………………..94

# **<span class="underline">LIST OF FIGURES</span>**

Figure 1. System Architecture and Design of DEVI 2

Figure 2. DEVI interactive processes 3

Figure 3. Flow diagram of the face recognition system and Overview of the DEVI chatbot 3

Figure 4. (a) The main page of DEVI GUI (b) Page to access DEVI chatbot in audio or text mode 4

Figure 5. DEVI the Robot Receptionist 5

Figure 6. Talking Receptionist Robot Methodology 6

Figure 7. System Architecture 7

Figure 8. Proposed Flow Diagram 7

Figure 9. Proposed Flow Chart 8

Figure 10. Voice Input Platform 8

Figure 11. HR Prototype 9

Figure 12. Flow Diagram of Face Recognition 10

Figure 13. Flow Chart of Speech Synthesis and Recognition 10

Figure 14. HR Robot Reply Section 10

Figure 15. Pepper 11

Figure 16. ARMAR II 11

Figure 17. BHR-5 12

Figure 18. People Bot Mobile Platform 13

Figure 19. People Bot SolidWorks Model (simplified) 14

Figure 20. Torso Conceptual Design 1 14

Figure 21. Torso Conceptual Design 2 15

Figure 22. Torso Conceptual Design 3 15

Figure 23. Torso Conceptual Design 4 (including arm) 16

Figure 24. Initial design exploded view 16

Figure 25. Initial design lateral view 17

Figure 26. Upper Torso (Half) 17

Figure 27. Lower Torso (Half) 17

Figure 28. Components Placement Plate 18

Figure 29. Arms Accessories Placement Plate 18

Figure 30. Neck Support 18

Figure 32. Tablet Holder 18

Figure 33. Plate to be mounted on People Bot Platform 19

Figure 34. Cover/Top Plates of Torso Assembly 19

Figure 35. Structure to be Mounted on the People Bot Platform Before Plates 19

Figure 36. Final design internal view 20

Figure 37. Final Design Excluding Torso 20

Figure 38. Final Design complete assembly on Peoplebot Mobile Platform 21

Figure 39. Head 22

Figure 40. Neck 22

Figure 41. Head Assembly 23

Figure 42. Neck 23

Figure 43. Half Head Manufactored 24

Figure 44. Half Head Assembly 24

Figure 45: Head Overview 25

Figure 46. Force Value 26

Figure 47. Force Application 27

Figure 48. Force Value for Tablet 27

Figure 49. Force application (for Tablet placement) 27

Figure 50. Fixed Value for Support 28

Figure 51. Fixed Support Application 28

Figure 52. Generating Mesh (Default size) 28

Figure 53. Total Deformation 29

Figure 54. Normal Elastic Strain 29

Figure 55. Equivalent Elastic Strain 30

Figure 56. Equivalent (von mises) Stress 30

Figure 57. Normal Stress 30

Figure 58. Total Deformation 31

Figure 59. Normal Stress 31

Figure 60. Initial Mesh 32

Figure 61. Global Refinement 33

Figure 62. Normal Stress 33

Figure 63. Global Refinement 34

Figure 64. Normal Stress 34

Figure 65. Convergence History for Static Torso 35

Figure 66. Meshing 36

Figure 67. Static Structure 36

Figure 68. Total Deformation 37

Figure 69. Equivalent Elastic Strain 37

Figure 70. Normal stress 37

Figure 71. Equivalent Stress 38

Figure 72. Explicit Dynamics for Torso Collision 38

Figure 73. Equivalent Elastic Strain for Torso Collision 39

Figure 74. Velocity vs. Max Stress 39

Figure 75. Base Plates (Mild Steel Laser Cutting and Welding) 45

Figure 76. Base plates (Transparent Acrylic Laser Cutting) 45

Figure 77. Half Component Plate 3D printed (bottom) 46

Figure 78. Half Assembly of Top 46

Figure 79. Component Support Plate 46

Figure 80. Support for Tablet Holder 47

Figure 81. Lower Torso 47

Figure 82. Tablet Holder 47

Figure 83. Head Support Plate/Top Plate 48

Figure 84. Front Torso Parts 48

Figure 85. PeopleBot 49

Figure 86. Arrangement of Parts of Torso 50

Figure 87. Torso Before Mounting 50

Figure 88. Torso Assembled 51

Figure 89. Training Results 53

Figure 90. Training Loss Accuracy 54

Figure 91. CNN Confusion Matrix 55

Figure 92. Sequential Layers 55

Figure 93. Precision Recall Flow 56

Figure 94: Surprised Male 56

Figure 95. Surprised Woman 57

Figure 96. Sad Male 57

Figure 97. Sad Woman 57

Figure 98. Neutral Male 57

Figure 99. Neutral Woman 58

Figure 100. Disgusted Male 58

Figure 101. Disgusted Woman 58

Figure 102. Happy Male 58

Figure 103. Happy Woman 59

Figure 104.Fear Male 59

Figure 105. Fear Woman 59

Figure 106. Angry Male 59

Figure 107. Angry Woman 60

Figure 108. Raspberry Pi 4B, 8Gb Ram 60

Figure 109. Snippet for TTS 66

Figure 110. Snippet for STT 67

Figure 111. Python Chatbot Development 69

Figure 112. Aspects 70

Figure 113. Implementation 72

Figure 114. Final Integration 77

Figure 115. Gender and Emotions 1 86

Figure 116. Gender and Emotions 2 87

Figure 117. Gender and Emotions 3 88

Figure 118. Main 1 88

Figure 119. Main 2 89

Figure 120. Main 3 89

Figure 121. Web1 90

Figure 122. Web2 90

Figure 123. Web3 91

**<span class="underline">LIST OF TABLES</span>**

Table 1. Technical Specifications of Nao………………………………………………………… 11

Table 2. Summary Of Motion, Joint Type, And Range of Rotation……………………………… 11,12

Table 3:  Summary Of Rubex Test………………………………………………………………... 16

Table 4. Action Units………………………………………………………………………………30

Table 5. Robot Orientation Data……………………………………………………………………31

> .
> 
> **<span class="underline">LIST OF SYMBOLS</span>**
> 
> *A* acceleration
> 
> *M* mass
> 
> *α* incidence angle
> 
> Ω resistance

**<span class="underline">Chapter 1 – INTRODUCTION</span>**

1.  **Overview:**

> The project undertakes the challenge of augmenting the communicative abilities and emotional engagement of humanoid robots, thereby enriching the user experience. With meticulous attention to detail, the platform is meticulously designed to incorporate key components essential for effective human-robot interaction. These components include a sophisticated mechanical structure encompassing a torso, head, and neck, facilitating fluid movement and interaction. Furthermore, a Python-based speech synthesis module is integrated, empowering the robot to comprehend and articulate information from PDF files, thereby enhancing its communicative repertoire.
> 
> Moreover, the project leverages deep learning methodologies to imbue the robot with perceptual capabilities, enabling it to discern scenes and detect human motion. Through the fusion Stereo camera using an Extended Kalman filter, the platform can accurately interpret its surroundings and respond with appropriate emotional expressions. This integration extends to a web application interface, seamlessly connecting users with the robot and enabling real-time interaction through spoken responses.
> 
> By synthesizing these components, the developed humanoid assistive robotic platform transcends traditional boundaries, offering enhanced communicative abilities, emotional engagement, and interaction capabilities. This thesis seeks to contribute to the burgeoning field of assistive robotics, driving advancements in human-robot interaction and fostering more intuitive and natural interactions between humans and robots across various contexts.

**<span class="underline">Chapter 2 - LITERATURE REVIEW</span>**

**2.1 DEVI: Open-source Human-Robot Interface for Interactive Receptionist System \[1\]**

The project explores the potential of socially interactive robots (SIR) and socially assistive robots (SAR) to enhance reception services, addressing the limitations often associated with human receptionists. It underscores the need for low-cost and user-friendly robot receptionists with open-source intelligence cores, aiming to overcome the challenges of current expensive solutions.

**2.1.1 System Architecture and Design**

The DEVI robot receptionist system features a modular architecture, enabling the isolation of individual modules and the seamless addition of new features. It stands out with a unique proximity sensing necklace, offering a 180-degree field of vision for effectively engaging with individuals approaching from various directions. This proximity map also optimizes the utilization of facial recognition processes, initiating recognition only when a person is in proximity. DEVI operates in three primary interaction scenarios: known-person identification, detection of unknown individuals, and handling false positives.

![](FYP_1/media/image3.png)

Figure 1. System Architecture and Design of DEVI

**2.1.1.1 Hardware Layer**

The hardware layer of DEVI comprises several key components. The proximity detection unit employs five VL53L0X time-of-flight sensors, creating a proximity map to track nearby individuals efficiently. The limb actuator control system manages the robot's hands for providing directional guidance, and the neck-motion control unit controls head movement. The main controller, an ATmega2560 microcontroller, interfaces with various components and maintains communication with the Linux host computer.

![](FYP_1/media/image4.png)

Figure 2. DEVI interactive processes: (a) Main Process 1- Person detection and identification (b) Main Process 2- User query services

**2.1.1.2 Robot Intelligence Core**

DEVI's intelligence core consists of three primary components: a face recognition system with a dynamic database, a chatbot, and a human-machine interface (HMI). The face recognition system relies on deep learning to identify known individuals and uses dynamic data to adapt and improve over time. The chatbot facilitates communication with users, responding to inquiries and offering information. The HMI includes a graphical user interface and Google Assistant integration, allowing users to interact with DEVI via both text and voice inputs.

![](FYP_1/media/image5.png)![](FYP_1/media/image6.png)

Figure 3. Flow diagram of the face recognition system and Overview of the DEVI chatbot

![A screenshot of a computer Description automatically generated](FYP_1/media/image7.png)

Figure 4. (a) The main page of DEVI GUI (b) Page to access DEVI chatbot in audio or text mode

The project aims to create a cost-effective and highly adaptable robot receptionist platform that can significantly enhance reception services, offering both practicality and customization for various applications. The open-source nature of DEVI allows users to tailor the system to their specific requirements, making it a versatile solution for interactive receptionist needs.

**2.1.2 Experiments and Results**

This section presents the findings from experiments conducted with the DEVI robot receptionist system.

General Specifications: DEVI has a weight of 9.9 kg, stands at a height of 110 cm, operates at a maximum current consumption of 1.32 A, with an idle current consumption of 0.18 A, and an operating voltage of 24 V.

Person Proximity Detection Unit: A linear recursive exponential filter is used to smooth noisy distance measurements from TOF sensors, with an optimal smoothing factor of 0.1 determined.

Face Recognition System: DEVI's face recognition system achieves a performance accuracy of 97.7% when tested against the "Faces in the Wild" dataset. It uses a camera with a 1.3 Megapixel image sensor.

Speech Recognition, Synthesis, and NLP: The NLP module operates at an accuracy of 91.7% in text mode, with potential for further improvement through training. In audio mode, accuracy varies from 77% to 50% due to the influence of ambient noise, suggesting the need for noise-cancellation solutions.

![](FYP_1/media/image8.png)

Figure 5. DEVI the Robot Receptionist

**2.2 Talking Receptionist Robot \[2\]**

The study emphasizes how important robots are to simplify daily living by lowering human error rates, saving time, and minimizing human effort. Robots have the potential to be extremely important in helping people with everyday tasks and providing services, such as serving as receptionists. The primary goal of this article is to develop a robot receptionist for college front offices, with a focus on integrating capabilities like speech and face recognition. These skills are essential for face-to-face communication to work well since voice carries important social cues such as gender, age, personality, emotional state, and origin. The study also emphasizes how crucial face detection is for deciphering body language and how the robot's capacity to identify non-verbal clues and sign language makes it accessible for people with difficulty speaking.

**2.2.1 Methodology**

There are multiple crucial steps in the Talking Receptionist Robot's technique. The robot recognizes a person when they get close and evaluates their body language and any sign language they may be using. It then extends a cordial greeting to the individual and shows it on a little LED screen. The robot then uses facial recognition to determine whether the visitor is returning or a first-time one. When a user logs in for the first time, the robot asks them for their name and records it in its database. The robot remembers guests' names and provides individualized support. The robot also has a specialized microphone for recording verbal input, which is likewise saved in the database. When a visitor wants to schedule a meeting with the college principal, the robot sends an inquiry message and a picture of their face to the relevant staff. The robot gives directions to guests and shows a brief route map on the LED screen to assist them even more.

![](FYP_1/media/image9.png)

Figure 6. Talking Receptionist Robot Methodology

**2.2.2 Experiments and Results**

The facial recognition and voice interaction skills of the receptionist robot were the main subjects of the experimentation. Face recognition performed remarkably well, allowing the robot to discriminate between people. The system's usefulness was demonstrated by comparing its performance to a pre-existing dataset. The results of voice interaction showed remarkable text mode accuracy, while ambient noise affected audio mode performance. The identification of difficulties in accurately transcribing audio questions in noisy contexts points to the necessity of noise-cancellation technologies. To sum up, the study tested the robot's ability to recognize faces for customized interactions. Nevertheless, difficulties with audio interactions in noisy environments were noted, highlighting the necessity of additional improvements to maximize voice capabilities for this flexible receptionist.

**2.3 Implementation of Voice Based Home Automation System Using Raspberry Pi\[3\]**

The goal of the project is to utilize an Android application and a Raspberry Pi to create a voice-activated home automation system. The objective is to develop an approachable voice command system for electronic device control, improving accessibility for the elderly and disabled. "Home Control," an Android app, is made for voice command-based device operations, registration, and login.

**2.3.1 System Design and Architecture**

The Raspberry Pi 3 model B, which has 1 GB of RAM and several peripherals, sits at the heart of the system. The Raspberry Pi is powered by a 5V power supply that is regulated to 3.3V as part of the architecture. Voice commands are wirelessly sent to the Raspberry Pi for additional processing from an Android handset. The operating system Ubuntu Mate and Java JDK 1.6 for front-end development are both included in the software stack. Python, MySQL, PHP, and Android Studio are essential software tools that make the system work. The system's hardware consists of a 15-inch VGA color monitor, a 16GB Class 10 SD card, a Raspberry Pi 3 model B, and other accessories. In addition, the display is connected via an HDMI to VGA converter.

![](FYP_1/media/image10.png)

Figure 7. System Architecture

**2.3.1.1 Proposed System**

The proposed system comprises a voice-activated system that transfers signals based on filtered spoken commands by using the GPIO pins of a Raspberry Pi. The user interface, which allows for remote connection with the Raspberry Pi, is an Android application. Users can run a variety of home appliances with the system thanks to Wi-Fi operation.

![](FYP_1/media/image11.png)

Figure 8. Proposed Flow Diagram

**2.3.1.2 Proposed System Workflow**

Starting the "Home Control" Android app, joining a specified Wi-Fi network, and registering are the first steps in the process. Users can operate individual appliances with voice commands after logging in. The Android handset translates voice commands into text, which is then sent to the Raspberry Pi. The Raspberry Pi uses text matching to operate appliances. Then, users can log off.

![](FYP_1/media/image12.png)![A diagram of a flowchart Description automatically generated](FYP_1/media/image13.png)

Figure 9. Proposed Flow Chart

**2.3.1.3 Methodology**

Voice commands are translated into text via the "Home Control" Android app. Relay switches are used by the Raspberry Pi to control electronic devices and manage text matching. Wi-Fi is used by the system for communication, and it provides flexibility for future upgrades.

![A screenshot of a phone Description automatically generated](FYP_1/media/image14.png)

Figure 10. Voice Input Platform

**2.3.2 Experiment and Results**

The voice-activated home automation system was implemented effectively by the project. The accuracy of voice recognition, which was dependent on pronunciation and speech intelligibility, was approximately 90%. Different Android versions were supported by the "Home Control" app. The solution showed promise for building an all-inclusive smart home environment that could be accessed remotely through Android devices.

**2.4 Design Modeling and Fabrication of Human-Humanoid Robot Communication \[4\]**

The design of humanoid robots for personal support jobs in various businesses and residences is covered in the article utilizing 3D modelling software. Humanoid robots are designed to help the aged and sick by doing jobs that humans may find difficult. These robots must communicate with people in a variety of methods, including spoken language and physical contact. The study highlights several humanoid robot research projects, including studies on perceptual design, motion planning, and human-robot interaction. It also emphasizes how crucial it is to change society to lower building costs and enable communication with humanoid robots. The creation of a humanoid robot prototype with sensors and the capacity to detect human interactions is described in the study. The prototype can react intelligently since it has a microcontroller unit, multiple cameras, and numerous sensors. Virtual Remote Control (VRC) is a remote mode of operation that allows the robot to converse.

**2.4.1 System Architecture and Design**

The system workflow for creating and modelling a humanoid robot (HR) is described in this study. It goes into using ProE to include complex shapes, Auto-CAD for 3D modelling, and hardware design tools like Arduino and NEC microcontrollers for multitasking. The software consists of embedded C programs, C coding, and Python. The design includes a 120GB solid-state hard drive and a 64-bit microprocessor with 4GB of RAM.

![](FYP_1/media/image15.png)

Figure 11. HR Prototype

**2.4.1.1 Methodology**

Supervised machine learning algorithms are used as part of the process. Speech synthesis and Hidden Markov Models (HMM) are used to convert modulated audio into analogue signals for speech recognition. Eigen faces are used in face identification, and Principal Component Analysis (PCA) is used to reduce dimensions. Python and the Open-Source Computer Vision Library (OSCVL) are used in speech recognition.

![](FYP_1/media/image16.png)

Figure 12. Flow Diagram of Face Recognition

![A diagram of a speech recognition Description automatically generated](FYP_1/media/image17.png)![A diagram of a process Description automatically generated](FYP_1/media/image18.png)

Figure 13. Flow Chart of Speech Synthesis and Recognition

![A diagram of a process Description automatically generated](FYP_1/media/image19.png)

Figure 14. HR Robot Reply Section

**2.4.2 Results and Discussion**

The Humanoid Robot (HR) that is described in the study performs admirably. With a 95% accuracy rate, the facial recognition technology prioritizes security. Pronunciation and microphone quality affect the accuracy of speech recognition. HR maintains a user database with varying degrees of access to perform a variety of tasks. We closely analyze response times and latency in remote access.

In conclusion, the HR prototype model shows itself to be a useful interactive tool with possible uses in defense and human aid. The outcomes demonstrate how accurate and precise it is at facial recognition for security reasons, which makes it a useful tool for enhancing human-machine interactions in practical settings.

**<span class="underline">Chapter 3 – HARDWARE METHADOLOGY</span>**

**3.1 Introduction**

The methodology chapter of this thesis acts as a guide that shows the logical and orderly approach used to develop HARP – from its theoretical formulation right down to its actual implementation and testing. So, the aim of this chapter is to present discussion and explanation of processes and methods that were used during the design process and throughout the idea and concept creating stages, and the final assembling of the product.

**3.1.1. Overview of design process**

The design process for the Humanoid assistive robotic platform (HARP) started by doing deep research into potential designs and their functionality, design efficiency and visual appeal. For this, the major focus was to look for designs of robotic platforms that were being used for similar purposes commercially. Robots such as Pepper, etc. gave great inspiration for design.

Since a movable people Bot platform was available to the project to serve as the source of mobility of the platform, the design process was focused towards designing the main torso of the body and head. The design had to encompass major components within the torso such as the battery packs, microcontrollers etc. while ensuring visual appeal and stability for the robot overall. Many iterative design cycles were conducted, all the while incorporating modifications based on feedback and evaluations.

After completing the design, the next step was the structural analysis using ANSYS software to evaluate the strength and overall stability of the design using the selected materials. This analysis gave valuable insights to the possible points of weaknesses and helped to the further refinement of the design.

**3.2. Design conceptualization**

**3.2.1. Initial conceptual phase**

During the initial step of conceptualization of the humanoid assistive robotic platform (HARP), a conceptual model was developed in terms of certain factors that were seen as critical to the design. The goals in the initial stages were to develop a robot about five feet tall and easily integrated into the existing people Bot chassis for installation. Drawing inspiration from research papers and commercially available robots being used in medical environments for effective human interaction, the design process started.

![A collage of a robot holding a sign Description automatically generated](FYP_1/media/image20.png)

Figure 15. Pepper

![A blue and silver robot Description automatically generated](FYP_1/media/image21.png)

Figure 16. ARMAR III

![A robot with a helmet Description automatically generated](FYP_1/media/image22.png)

Figure 17. BHR-5

**3.2.2. Methodology to generate design concepts:**

The approach used in developing the design concepts was complex, taking into account various aspects of the potential product including functionality, appearance, and manufacturability. As the concept was created, different theories were deployed, and models were created and improved until a reasonable combination of feasibility and creativity was achieved.

Special emphasis was placed throughout the conceptualization phase to provide the robotic platform with some form of humanoid appearance while at the same time integrating equipment and components in a manner which is more cost effective and functional. This entailed paying attention to confronting aesthetic appearance and the possibility to use techniques and components to achieve an aesthetic appearance for the design. At the same time, design solutions were offered to increase the internal layout for the required hardware and electronics while keeping the appearance at a sufficient level.

**3.3. Solidworks modelling**

**3.3.1. Peoplebot Solidworks model**

Since the torso was to be assembled and mounted on the Peoplebot platform, the first requirement to get started with the design phase was a design model for Peoplebot platform. Despite research, such a model could not be found. Hence arose the need to create a design model for the platform by taking actual dimensions from the available peoplebot platform. This proved a tedious task with more chances of errors in measurement and subsequent design. Great attention was paid to maintaining accuracy in the dimensions of the model.

The Model designed on SolidWorks was simplified to just incorporate the main parts of the platform that would have a direct relation with the subsequent torso design. Hence the LIDAR sensor platform was not designed.

![](FYP_1/media/image23.png)

Figure 18. PeopleBot Mobile Platform

![A drawing of a mechanical device Description automatically generated](FYP_1/media/image24.png)

Figure 19. PeopleBot Solidworks Model (simplified)

**3.3.2. Concept Exploration**

Based upon this model of the peoplebot, a number of design ideas were then pursued in an attempt to design the torso section of the humanoid robot. The first torso model included a small number of parts, where a major part of the torso consisted of a front and a back torso connected by a few engagements. The internal layout was some component placement sections in the torso itself.

![A drawing of a metal arm Description automatically generated](FYP_1/media/image25.png)

Figure 20. Torso Conceptual Design 1

![A grey object with legs Description automatically generated](FYP_1/media/image26.png)

Figure 21. Torso Conceptual Design 2

![A drawing of a black object Description automatically generated](FYP_1/media/image27.png)

Figure 22. Torso Conceptual Design 3

![A grey robot with a long handle Description automatically generated with medium confidence](FYP_1/media/image28.png)

Figure 23. Torso Conceptual Design 4 (including arm)

**3.3.3. Iterative design process:**

The design process was iterative, based on constant evaluations and feedback. The two torso parts modified into 4 with adjustment points and inclusions of holes and keys for attachment. Instead of making compartments in the torso structure itself, it was decided to include apportions in the middle to accommodate essential components such as the jetson nano processor, speakers, and motor placements for future arm designs. The torso was now just a shell to provide visual appearance as well as a safe compartment for the components placed inside. Each modification had sought to increases functionality, integrate, and maximize internal space, and increase the designs ergonomic features.

![A drawing of different shapes of furniture Description automatically generated with medium confidence](FYP_1/media/image29.png)

Figure 24. Initial design exploded view

![A computer generated image of a machine Description automatically generated](FYP_1/media/image30.png)

Figure 25. Initial design lateral view

**3.4. Parts:**

![A black and grey object with holes Description automatically generated with medium confidence](FYP_1/media/image31.png)

Figure 26. Upper Torso (Half)

![A grey curved object with three black legs Description automatically generated](FYP_1/media/image32.png)

Figure 15. Lower Torso (Half)

![A green rectangular object with holes Description automatically generated](FYP_1/media/image33.png)

Figure 16. Component Placement Plate

![A green metal bar with a long beam Description automatically generated with medium confidence](FYP_1/media/image34.png)

Figure 29. Arms Accessories Placement Plate

![A rectangular object with a hole Description automatically generated](FYP_1/media/image35.png)

Figure 30. Neck Support

![A blue metal shelf with holes Description automatically generated](FYP_1/media/image36.png)

Figure 17. Tablet Holder Attachment Plate

![A blue metal bracket with screws Description automatically generated](FYP_1/media/image37.png)

Figure 32. Tablet Holder

![A black object with holes Description automatically generated](FYP_1/media/image38.png)

Figure 33. Plate to be Mounted on Peoplebot Platform

![A grey object with a white circle Description automatically generated with medium confidence](FYP_1/media/image39.png)

Figure 18. Cover/Top plates of Torso Assembly

![A drawing of a rectangular object Description automatically generated](FYP_1/media/image40.jpg)

Figure 35. Structure to be Mounted on the Peoplebot Platform Before Plates

**3.4.1 Final Torso Assembly:**

![A drawing of a structure Description automatically generated](FYP_1/media/image41.png)

Figure 36. Final Design Internal View

![A drawing of a machine Description automatically generated](FYP_1/media/image42.png)

Figure 37. Final Design Excluding Torso

![A drawing of a machine Description automatically generated](FYP_1/media/image43.png)

Figure 38. Final Design Complete Assembly on Peoplebot Mobile Platform

**3.5. Design and implementation of Head**

In the construction of HARP, the head and neck junction are important parts of the system architecture. These components not only provide necessary hardware in units like the stand, prominently for the display screen and the camera but also use a lot play the roles of proper aesthetics and utility in the structure of the robot. In this section, the author presents the design and development phase using the SolidWorks software, the creation of the components through 3D printing, and the application of ANSYS in static analysis of the created components in order to check the structural stability.

**3.5.1. Design Process**

The design process consists of the initial concept based on requirements and the solidworks design and manufacturing.

**3.5.1.1. Initial Concept and Requirements**

The head and neck of HARP were conceptualized to fulfill several requirements: The head and neck of HARP were conceptualized to fulfill several requirements:

**3.5.1.2. Functional Integration**

The head must provide clear space for adapting a 7 inches display screen and an Intel camera that facilitate the interaction of the users and the perception of the environment.The components must be capable of surviving the mechanical loads while the automobile is in use.  
  
**3.5.2 SolidWorks Design**

> The initial step towards designing was prototyping by using a CAD modeling tool known by the name of SolidWorks. The head and neck were designed in such a manner that they are built like two distinct segments that are able to fit into each other this made the construction easier. Key features of the design include:

  - > **Screen Housing**

> The front side of the headunit contains a groove specifically for hosting a 7 inch display screen. This positioning guarantees that the side of the screen with the functional buttons is reachable for the user.

  - > **Camera Mount**

> One attaches firmly on the Intel camera has been placed on the bottom part of the screen in such a way that the camera will be free to have a clear vision for detecting the emotion of users and scanning the environment.

  - > **Cable Management**

> Concisely, to accommodate the hoses and ensure the cables connecting the screen and camera to the rest of the system remain secure, slot is formed from neck to connect with lower torso design.

  - > **Mounting Points**

> The neck part is connected fixed with the body of robot through nut and bolts.  
> As depicted in the figure, the final design of the head and neck in SolidWorks is as shown below; Figure 52,53 and 54 enthusiasts for a better understanding.

![](FYP_1/media/image44.png)

Figure 39. Head

![](FYP_1/media/image45.png)  
Figure 40. Neck

![](FYP_1/media/image46.png)

Figure 41. Head Assembly

**3.5.3. Manufacturing of head**

The printed parts were then mounted to check for their fit as well as how they were going to function.

> In figure 55 we have the post processed 3D printed components acquired from the fabrication stage.

![A hand holding a grey plastic object Description automatically generated](FYP_1/media/image47.jpeg)

Figure 42: Neck

![A hand holding a white object Description automatically generated](FYP_1/media/image48.jpeg)

Figure 43 Half Head Manufactored

![A grey plastic pipe on a wood surface Description automatically generated](FYP_1/media/image49.jpeg)

Figure 44. Half Head Assembly

**3.5.4. Final Assembly**

inch display and intel camera is mounted on head with neck support at bottom which ensure correct alingment integrity of the robot’s head. Below is an image showcasing the assembled head and neck 7

![A machine with a screen on top Description automatically generated](FYP_1/media/image50.jpeg)

Figure 45: Head Overview

**3.6 Design Challenges and Strategies:**

Some of the difficulties observed during the design stage were element size, the restriction to the initial limitation, and the issue of how to attach elements properly. Some of the strategies used to overcome these challenges include changes of the placements of the bot components through an integrated method, the division of the torso into different plates in order to increase its stability and strength as well as the development of mounting plates in order to merge seamlessly with the peoplebot.

**3.7. Design Validation:**

After completing the design, validation of the model was necessary before moving to the manufacturing phase. Static structural analysis in ANSYS refers to the FEA that is used to determine the response of a structure when it is subjected to loads. This analysis allows one to anticipate deformation, stress as well as strain in the particular component or assembly so that the design contains the necessary safety and performance characteristics.

**3.7.1. Finite Element Analysis (FEA):**

FEA is defined as a numerical approach, which predicts the responses of structures and materials under several physical fields. It decomposes the structure into small parts (mesh) and the physical equations are solved over all these parts with the results then combined to arrive at the global solution.

**3.7.2. Static Structural Analysis:**

This form of analysis assumes that the loads are gradually applied to the structure and therefore, the equilibrium state of the structure is static with no temporal considerations.

Some of the output solutions obtained are displacements, strains, stresses, and reaction forces.

**3.7.3. ANSYS Workbench:**

ANSYS Workbench is a very friendly user interface in which all the analysis tools can be found. It gives a graphical user interface for defining and solution of FEA problems as well as post processing.

Virtual simulations were done using ANSYS software. The FEA Analysis was done only on the parts the composed the basic layout of the robot and were bearing most of the weight of the robot. The torso outside parts just acted as a shell since they had a 3mm thickness and were not holding any component by themselves.

Stress, strain and deformation analysis were performed while changing the mesh sizes to ensure mesh quality and refined results. Computing ANSYS Analysis excluding torso since major weight is centered. And the torso is just a shell bearing minimum weight. Since most of the component weight will be centered on the middle portion, hence force is applied for analysis on the middle 3D printed structure.

**3.8 Static Analysis Using ANSYS**

This analysis assists in finding out the flaw that may be present and to make sure that the design is strong enough to handle the forces that are likely to be applied on it when in use.**  
  
3.8.0.1. Simulation Setup:**

The static analysis was carried out using ANSYS software – a reliable software in solving engineering analysis. The setup involved the following steps:The setup involved the following steps:

  - > **Importing the Model**:

The head and neck part models created in solidworks were exported and imported into ANSYS software.

  - > **Material Properties:**

Density, Young’s modulus, and Poisson’s ratio of PLA were determined as follows: Density = 1.24 g/cm³; Young’s modulus = 3.75 GPa; Poisson ratio = 0.34.

Acrylic/PMMA

  - > **Boundary Conditions**:

Proper boundary conditions were used to model the connections and limited parts of the model.

  - **Mesh density:** The infill pattern is honeycomb and 20% infill density. Hence adjusting the material properties to account for infill density. It assumes a homogenized material behaviour.

**3.8.0.2. Analysis Process:**

Static forces and moments to account for the screen and camera weight were applied together with any operational forces resulting from motion.

![](FYP_1/media/image51.png)

Figure 46. Forces Value

![A computer generated model of a table Description automatically generated](FYP_1/media/image51.png)

Figure 47. Force Application

Since the tablet will be placed at a separate holder, it would have a different effect on the assembly. Hence arose the need to apply a small separate force on the holder.

![](FYP_1/media/image52.png)

Figure 48. Force Value for Tablet

![A computer generated model of a table Description automatically generated](FYP_1/media/image52.png)

Figure 49. Force application (for Tablet placement)

The fixed support is applied for analysis purposes at the base plate that will directly be in contact with the peoplebot platform. Since this analysis is structural, hence assuming for now that the peoplebot platform is stationary.

![A computer generated image of a table Description automatically generated](FYP_1/media/image53.png)

Figure 50. Force Value for Support

![A computer generated image of a table Description automatically generated](FYP_1/media/image53.png)

Figure 51. Fixed Support Application

![](FYP_1/media/image54.png)  
Figure 52. Generating Mesh (Default size)

Reducing the mesh size subsequently to gain accurate and better results and convergence.

**3.8.1. Results:**

![A multicolored object with a hole Description automatically generated](FYP_1/media/image55.png)![A multicolored object with a hole Description automatically generated](FYP_1/media/image55.png)

Figure 53. Total Deformation

![A yellow object with a hole in the middle Description automatically generated](FYP_1/media/image56.png)![A yellow object with a hole in the middle Description automatically generated](FYP_1/media/image56.png)

Figure 54. Normal Elastic Strain

![A blue object with a hole in the top Description automatically generated](FYP_1/media/image57.png)![A blue object with a hole in the top Description automatically generated](FYP_1/media/image57.png)

Figure 55. Equivalent Elastic Strain

![A blue object with a blue background Description automatically generated](FYP_1/media/image58.png)![A blue object with a blue background Description automatically generated](FYP_1/media/image58.png)

Figure 56. Equivalent (von mises) Stress

![A yellow and green object Description automatically generated](FYP_1/media/image59.png)![A yellow and green object Description automatically generated](FYP_1/media/image59.png)

Figure 57. Normal Stress

**3.8.2. Including the torso:**

**Results:**

The results show that the main weight is still centered towards the middle portion of the assembly i.e. the torso has not much effect on the mass distribution and stress concentration.

![](FYP_1/media/image60.png)

Figure 58. Total Deformation

![](FYP_1/media/image61.png)

Figure 59. Normal Stress

**3.8.3. Mesh convergence and importance:**

They assist to justify the simulation analysis results as accurate and dependable by checking the mesh convergence in ANSYS as a part of FE Analysis. It elevates the mesh density to a level were raising density values barely changes solutions, this determines that the solution has reached convergence.

**3.8.3.1.** **Initial Setup**

**• Define Geometry:** The best approach is to start with the creation of a geometry or use one that was developed in the previous project.

• **Assign Material Properties:** In the model you are going to develop all the properties of the model materials should be defined.

• **Apply Boundary Conditions:** These are the conditions applied in the assessment of the structures in the environment as follow:

**3.8.3.2. Initial Meshing**

• **Generate Initial Mesh:** Designate the initial set of standard values for an abdominal muscle. There are various approaches that can be used while meshing the structures which can be fully automatic, manually controlled or of high level one. While making a simulation, there are various choices which we have, but the first preference for a novice is the Automatic field setting.

> ![](FYP_1/media/image54.png)

Figure 60. Initial Mesh

**3.8.3.3. Run Initial Analysis**

**• Solve the Initial Mesh**: The process starts with the first Meshing requirements choice and then the analysis is performed to achieve the Meshing results.

**• Examine Results**: Depending on the type of post processing analysis you decide to conduct, it may include certain variables such as stresses, displacements or temperature fields.

**3.8.3.4. Mesh Refinement**

**• Refine the Mesh:** Thus, examine the mesh refinement in the smaller amount of metadata that is expressed by high gradient or large differences in the lists of results. This is often the place where values of stresses or other postulated response functions are at their maximum.

**• Global Refinement:** Coarsen the whole mesh and did it uniformly or use the number of iterations coarsen the whole mesh and did it uniformly**.**

**• Local Refinement:** Apply only in areas of interest such as zones that experience high stress or boast complicated material geometry**.**

![A computer generated image of a small table Description automatically generated with medium confidence](FYP_1/media/image62.png)

Figure 61. Global Refinement

**3.8.3.5. Re-run the Analysis.**

**• Re-solve with Refined Mesh:** Repeat the simulation with the new mesh parameterization that is improved according to the results.

• **Compare Results:** When using this refined mesh, compare the results obtained with results achieved from the previous mesh. Such criteria that can help to compare the results can be such unique values as the maximum stress, displacement, or other response values.

![A green and yellow vehicle Description automatically generated with medium confidence](FYP_1/media/image63.png)![A green and yellow vehicle Description automatically generated with medium confidence](FYP_1/media/image63.png)  
Figure 62. Normal Stress

**3.8.3.6. Convergence Check**

**• Check Convergence Criteria:** It is critical to find out if the results converged. This is mainly accomplished by comparing the relative change in the results at two consecutive meshes refinement level.

**• Convergence Plot**: As a simulation, make a graph of two variables, for example, maximum stress parallel to the number of elements. The answers ought to converge endlessly toward a distinct value with the refinement of the mesh**.**

**• Acceptance Criteria:** Agree on an acceptable level of change this means agreeing and setting the limit of how much change is acceptable between phases/iterations (for instance, change rate less than 5% between two refinements).

**3.8.3.7. More sophistication if needed**

**• Iterate as Needed:** If the results did not converge, the mesh needs to be updated and the analysis repeated with smaller and denser meshes until the results margin changes fall are within the accepted range**.**

![A green and blue object Description automatically generated](FYP_1/media/image64.png)![A green and blue object Description automatically generated](FYP_1/media/image64.png)

Figure 63. Global Refinement

![A yellow and green object Description automatically generated](FYP_1/media/image59.png)![A yellow and green object Description automatically generated](FYP_1/media/image59.png)

Figure 64. Normal Stress

**3.8.3.8. Final Validation**

  - **Verify with Theoretical/Experimental Data**: If available, compare your converged results with theoretical calculations or experimental data for validation

> ![A graph with a red line Description automatically generated](FYP_1/media/image65.png)

Figure 65. Convergence History for Static Torso

  - **Document the Process:** Document the mesh convergence analysis checks the numerous mesh options available, and the results obtained at each iteration finally the results obtained at the end of mesh convergence study.

  - **Adaptive Meshing:** AN lookup. Use the adaptive meshing techniques available in ANSYS, which can adapt the size of the mesh in a post-processing step based on error estimates obtained from the initial analysis.

  - **Element Quality:** Make sure that the mesh elements are fine in order to condone the numerical troubles. Scan for the underlying graphical items that have low AR or skewed images.

  - **Appropriate Element Types:** Some constructions may require using only point and straight line, others may require using other element types. For instance, employ higher order elements only in areas that demand enhanced accuracy levels.

Also, mesh convergence was done by changing the allowable value of change consequently and the following graph was generated which approached a nearly constant result of normal stress.

The results obtained gave an identification of stress concentration areas and optimization for performance and safety.

**  
**

**3.9 Results and Analysis**

**  
**The static analysis was able to show the assorted stresses and deformation that the head and neck would undergo.

  - **Stress Concentrations:** This study showed the stress concentration zones that were mainly near the mounting points as well as the margins of the screen housing**.**

  - **Deformation**: There was very little deformation under load and therefore one can conclude that the design of the structure can support the loads that are imposed to the attached structures.

From the above analysis, it was possible to make minor changes to the design after identifying the zones that experience high stress concentration. The stress distribution obtained is shown in Figures.

![A black square object with a nut Description automatically generated](FYP_1/media/image66.png)

> Figure 66. Meshing

![A screenshot of a computer Description automatically generated](FYP_1/media/image67.png)![A computer generated image of a device Description automatically generated with medium confidence](FYP_1/media/image68.png)

Figure 67. Static Structure

![A screenshot of a computer Description automatically generated](FYP_1/media/image69.png)![A 3d model of a light bulb Description automatically generated](FYP_1/media/image70.png)

Figure 68. Total Deformation

![](FYP_1/media/image71.png)![A close-up of a camera Description automatically generated](FYP_1/media/image72.png)

Figure 69. Equivalent Elastic Strain

![](FYP_1/media/image73.png)![A yellow object with a green square Description automatically generated](FYP_1/media/image74.png)

Figure 70. Normal Stress

![](FYP_1/media/image75.png)![A blue square object with a square object with a square object with a square object with a square object with a square object with a square object with a square object with a square object with a square Description automatically generated](FYP_1/media/image76.png)

Figure 71. Equivalent Stress

**3.10. Dynamic Analysis**

In the Dynamic analysis, the torso is made to collide with a concrete wall to analyse the possible effects of the collision on the torso. The torso is given an initial speed of 0.3 m/s since the average speed of people Bot platform ranges between 0.3 – 0.5 m/s. On striking the wall, the equivalent (von mises) stress is centered on the front centre of torso and spreading symmetrically and in less proportion to the other parts. Similar is the case with equivalent elastic strain. It spreads symmetrically and within acceptable range of the allowable strain.

![A computer generated image of a tooth Description automatically generated](FYP_1/media/image77.png)![A computer generated image of a tooth Description automatically generated](FYP_1/media/image77.png)

Figure 72. Explicit Dynamics for Torso Collision

![A blue and red globe Description automatically generated](FYP_1/media/image78.png)![A blue and red globe Description automatically generated](FYP_1/media/image78.png)

Figure 73. Equivalent Elastic Strain for Torso Collision

Increasing the velocities and analysing the max stress produced in the torso structure correspondingly to get an approximation of the collision analysis of the torso and concrete wall.

Table 1: Velocity with stress

<table>
<thead>
<tr class="header">
<th><p>Velocity</p>
<p>(m/s)</p></th>
<th>Max. stress (Pa)</th>
<th>Time(s)</th>
<th>Min. stress (Pa)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>0.3</td>
<td>2.6208e6</td>
<td>3.0003e-0.003</td>
<td>3762.8</td>
</tr>
<tr class="even">
<td>0.4</td>
<td>3.5332e6</td>
<td>2.0002e-003</td>
<td>2748</td>
</tr>
<tr class="odd">
<td>0.5</td>
<td>4.5645e6</td>
<td>2.0002e-003</td>
<td>7257.7</td>
</tr>
<tr class="even">
<td>0.6</td>
<td>5.3324e6</td>
<td>2.0001e-003</td>
<td>7480.9</td>
</tr>
<tr class="odd">
<td>0.7</td>
<td>6.2476e6</td>
<td>2.5003e-003</td>
<td>11744</td>
</tr>
</tbody>
</table>

![A graph with a line going up Description automatically generated](FYP_1/media/image79.png)

Figure 74. Velocity vs. Max Stress

**3.11. Manufacturing**

The Designed parts were different in terms of use and complexity so different methods of manufacturing and materials were considered based on the requirements and their effectiveness.

**3.11.1.** **Available materials**

Table 2: Materials with their properties

<table>
<thead>
<tr class="header">
<th>Material</th>
<th>Properties</th>
<th>Pros</th>
<th>cons</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Mild Steel</td>
<td><p>Composition: Primarily iron with a small amount of carbon (typically 0.05-0.25%).</p>
<p>Density: ~7.85 g/cm³.</p>
<p>Tensile Strength: ~400-550 MPa.</p>
<p>Modulus of Elasticity: ~210 GPa.</p></td>
<td><p><strong>Cost-Effective</strong>: Inexpensive compared to many other metals.</p>
<p><strong>Ductility</strong>: Easily shaped and formed.</p>
<p><strong>Weldability</strong>: Excellent for welding and fabrication.</p>
<p><strong>Recyclability</strong>: Highly recyclable, reducing environmental impact.</p></td>
<td><p><strong>Corrosion</strong>: Prone to rust and corrosion if not properly protected.</p>
<p><strong>Strength</strong>: Lower strength compared to high-carbon steels or alloys.</p>
<p><strong>Weight</strong>: Heavier than many other structural materials like aluminum or composites.</p></td>
</tr>
<tr class="even">
<td>Carbon Fiber</td>
<td><p>Composition: Carbon fibers embedded in a polymer matrix.</p>
<p>Density: ~1.6 g/cm³.</p>
<p>Tensile Strength: ~3,500 MPa.</p>
<p>Modulus of Elasticity: ~230-600 GPa.</p></td>
<td><p><strong>High Strength-to-Weight Ratio</strong>: Extremely strong yet lightweight.</p>
<p><strong>Stiffness</strong>: High modulus of elasticity provides rigidity.</p>
<p><strong>Corrosion Resistance</strong>: Does not rust or corrode.</p>
<p><strong>Thermal Stability</strong>: Good performance at a wide range of temperatures.</p></td>
<td><p><strong>Cost</strong>: Expensive to produce and manufacture.</p>
<p><strong>Brittleness</strong>: Can be brittle and prone to cracking.</p>
<p><strong>Manufacturing Complexity</strong>: Requires specialized manufacturing processes and equipment.</p>
<p><strong>Recyclability</strong>: More challenging to recycle compared to metals.</p></td>
</tr>
<tr class="odd">
<td>Acrylic (PMMA)</td>
<td><p>Composition: Polymethyl methacrylate.</p>
<p>Density: ~1.18 g/cm³.</p>
<p>Tensile Strength: ~70 MPa.</p>
<p>Modulus of Elasticity: ~3.2 GPa.</p></td>
<td><p><strong>Transparency</strong>: Excellent optical clarity, often used as a glass substitute.</p>
<p><strong>Weather Resistance</strong>: Resistant to UV light and weathering.</p>
<p><strong>Lightweight</strong>: Lighter than glass and many other plastics.</p>
<p><strong>Easy to Fabricate</strong>: Can be easily cut, shaped, and bonded.</p></td>
<td><p><strong>Scratch Resistance</strong>: Can be easily scratched compared to glass.</p>
<p><strong>Impact Resistance</strong>: More brittle and less impact-resistant than polycarbonate.</p>
<p><strong>Thermal Resistance</strong>: Low heat resistance, can warp or melt at high temperatures.</p></td>
</tr>
<tr class="even">
<td>PLA (Polylactic Acid)</td>
<td><p>Composition: Derived from renewable resources like corn starch.</p>
<p>Density: ~1.25 g/cm³.</p>
<p>Tensile Strength: ~50-70 MPa.</p>
<p>Modulus of Elasticity: ~3.5 GPa.</p></td>
<td><p><strong>Biodegradable</strong>: Eco-friendly and compostable under industrial conditions.</p>
<p><strong>Ease of Use</strong>: Low warping, easy to print, minimal odor.</p>
<p><strong>Surface Finish</strong>: Produces smooth and shiny surfaces.</p>
<p><strong>Low Printing Temperature</strong>: Typically prints at 180-220°C.</p></td>
<td><p><strong>Brittleness</strong>: More brittle compared to other 3D printing plastics.</p>
<p><strong>Heat Resistance</strong>: Poor thermal resistance; deforms at relatively low temperatures (~60°C).</p>
<p><strong>Moisture Absorption</strong>: Absorbs moisture from the air, which can affect print quality.</p></td>
</tr>
<tr class="odd">
<td>PET-G (Polyethylene Terephthalate Glycol)</td>
<td><p>Composition: A glycol-modified version of PET.</p>
<p>Density: ~1.27 g/cm³.</p>
<p>Tensile Strength: ~50-70 MPa.</p>
<p>Modulus of Elasticity: ~2 GPa.</p></td>
<td><p><strong>Strength and Durability</strong>: Stronger and more durable than PLA.</p>
<p><strong>Flexibility</strong>: More flexible and less brittle than PLA.</p>
<p><strong>Chemical Resistance</strong>: Resistant to water, chemicals, and impact.</p>
<p><strong>Ease of Printing</strong>: Low warping and good layer adhesion.</p></td>
<td><p><strong>Stringing</strong>: Can suffer from stringing issues during printing.</p>
<p><strong>Bed Adhesion</strong>: Requires a heated bed for optimal adhesion.</p>
<p><strong>Cost</strong>: Slightly more expensive than PLA.</p></td>
</tr>
<tr class="even">
<td>ABS (Acrylonitrile Butadiene Styrene)</td>
<td><p>Composition: A copolymer made from acrylonitrile, butadiene, and styrene.</p>
<p>Density: ~1.04 g/cm³.</p>
<p>Tensile Strength: ~40-50 MPa.</p>
<p>Modulus of Elasticity: ~2.3 GPa.</p></td>
<td><p><strong>Impact Resistance</strong>: Good toughness and impact resistance.</p>
<p><strong>Thermal Stability</strong>: Higher heat resistance than PLA (~100°C).</p>
<p><strong>Post-Processing</strong>: Can be easily sanded, machined, and painted.</p>
<p><strong>Strength</strong>: Stronger than PLA.</p></td>
<td><p><strong>Warping</strong>: Prone to warping and requires a heated bed and enclosure.</p>
<p><strong>Odor</strong>: Emits strong fumes during printing; requires good ventilation.</p>
<p><strong>Environmental</strong></p>
<p><strong>Impact</strong>: Non-biodegradable and less eco-friendly compared to PLA.</p></td>
</tr>
</tbody>
</table>

Each material has its unique advantages and limitations, making them suitable for different applications depending on the requirements such as strength, flexibility, durability, cost, and environmental impact.

**3.11.1. Selected materials**

The base plates that were to be mounted on the peoplebot robot, they were initially selected to be made of Mild steel employing the use of laser cutting and welding method. In the assembly phase however, it was found that the material was not suitable for the project and hence the parts were then made of acrylic sheets 5mm thick and transparent using laser cutting method.

Transparent and colored acrylic are two variations of acrylic (PMMA) that differ primarily in their appearance due to the presence of colorants in the latter.

All the other parts were 3d printed using PLA material.

**3.11.2. Available methods**

Table 3: Available methods

<table>
<thead>
<tr class="header">
<th>Methods</th>
<th>Details</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><ol type="1">
<li><p>CNC Machining</p></li>
</ol></td>
<td><p><strong>Process</strong>: Uses computer-controlled machines to remove material from a solid block (subtractive manufacturing).</p>
<p><strong>Materials</strong>: Wide range including steel, aluminum, brass, titanium.</p>
<p><strong>Advantages</strong>:</p>
<p>High precision and surface finish.</p>
<p>Suitable for both prototyping and production.</p>
<p>Works with a wide range of materials.</p>
<p><strong>Disadvantages</strong>:</p>
<p>Material waste can be high.</p>
<p>Complex geometries can be challenging and costly.</p></td>
</tr>
<tr class="even">
<td><ol start="2" type="1">
<li><p>Casting</p></li>
</ol></td>
<td><p><strong>Process</strong>: Molten metal is poured into a mold and allowed to solidify.</p>
<p><strong>Materials</strong>: Iron, steel, aluminum, brass, bronze.</p>
<p><strong>Advantages</strong>:</p>
<p>Economical for large quantities.</p>
<p>Capable of producing complex shapes.</p>
<p>Good mechanical properties.</p>
<p><strong>Disadvantages</strong>:</p>
<p>High initial tooling cost.</p>
<p>Surface finish and precision can be lower compared to machine.</p>
<p>Not suitable for low-volume production.</p></td>
</tr>
<tr class="odd">
<td><ol start="3" type="1">
<li><p>Forging</p></li>
</ol></td>
<td><p><strong>Process</strong>: Metal is shaped by compressive forces, usually using a hammer or press.</p>
<p><strong>Materials</strong>: Steel, aluminum, titanium, copper alloys.</p>
<p><strong>Advantages</strong>:</p>
<p>Excellent mechanical properties.</p>
<p>Strong and durable parts.</p>
<p>Suitable for high-stress applications.</p>
<p><strong>Disadvantages</strong>:</p>
<p>Limited to simpler shapes.</p>
<p>High tooling costs.</p>
<p>Requires significant post-processing.</p></td>
</tr>
<tr class="even">
<td>4. Sheet Metal Fabrication</td>
<td><p><strong>Process</strong>: Involves cutting, bending, and assembling flat metal sheets.</p>
<p><strong>Materials</strong>: Steel, aluminum, brass, copper.</p>
<p><strong>Advantages</strong>:</p>
<p>Cost-effective for thin parts.</p>
<p>High precision and repeatability.</p>
<p>Versatile for a wide range of applications.</p>
<p><strong>Disadvantages</strong>:</p>
<p>Limited to sheet materials.</p>
<p>Complex shapes can be challenging.</p></td>
</tr>
<tr class="odd">
<td>5. Injection Molding</td>
<td><p><strong>Process</strong>: Molten acrylic is injected into a mold and solidified.</p>
<p><strong>Materials</strong>: PMMA (Acrylic)</p>
<p><strong>Advantages</strong>:</p>
<p>High production efficiency for large quantities.</p>
<p>Excellent surface finish and detail.</p>
<p>Consistent and repeatable parts.</p>
<p><strong>Disadvantages</strong>:</p>
<p>High initial tooling cost.</p>
<p>Not economical for low-volume production.</p>
<p>Limited to relatively simple shapes.</p></td>
</tr>
<tr class="even">
<td>6. Laser Cutting and Engraving</td>
<td><p><strong>Process</strong>: Uses a laser to cut or engrave acrylic sheets.</p>
<p><strong>Materials</strong>: Acrylic sheets.</p>
<p><strong>Advantages</strong>:</p>
<p>High precision and clean edges.</p>
<p>Fast and efficient for sheet materials.</p>
<p>Versatile for custom designs and small batches.</p>
<p><strong>Disadvantages</strong>:</p>
<p>Limited to 2D shapes and flat sheets.</p>
<p>Thickness limitations based on laser power.</p></td>
</tr>
<tr class="odd">
<td><ol start="7" type="1">
<li><p>Thermoforming</p></li>
</ol></td>
<td><p><strong>Process</strong>: Acrylic sheets are heated until pliable, then formed over a mold.</p>
<p><strong>Materials</strong>: Acrylic sheets.</p>
<p><strong>Advantages</strong>:</p>
<p>Economical for large parts.</p>
<p>Capable of producing complex curves and shapes.</p>
<p>Good for medium to large production runs.</p>
<p><strong>Disadvantages</strong>:</p>
<p>Limited to thin-walled parts.</p>
<p>Surface detail and precision are lower than injection molding.</p></td>
</tr>
</tbody>
</table>

**  
**

Each of these manufacturing methods offers unique advantages and disadvantages, making them suitable for different applications depending on factors such as production volume, part complexity, material properties, and cost considerations.

**3.11.3. Selected Methods**

Methods selected were 3D printing, laser cutting and welding.

**3.11.1.1 Laser Cutting and Welding**

![](FYP_1/media/image80.jpeg)

Figure 75. Base Plates (Mild Steel Laser Cutting and Welding)

![A circular object with holes and writing on it Description automatically generated](FYP_1/media/image81.jpeg)

Figure 76. Base plates (Transparent Acrylic Laser Cutting)

**3.11.3.2 3D printing:**

Steps involved in preparing for 3d printing include exporting the model in a format compatible with the 3d printing software i.e. STL(Stereolithography) files, slicing the model using Cura (commonly used with ender 3 pro 3D Printer), configuring print settings and dividing large parts into smaller sections.

For PLA (Polylactic Acid), typically used temperatures are around 200-220°C for the nozzle and 50-60°C for the bed.

Support structures were required to prevent collapse during printing. To prevent warping, especially with larger parts, good bed adhesion was ensured.

![](FYP_1/media/image82.jpeg)

Figure 77. Half Component Plate 3D printed (bottom)

![A hand holding a white object Description automatically generated](FYP_1/media/image83.jpeg)

Figure 78. Half Component Plate 3D printed (top)

![A white frame on a table Description automatically generated](FYP_1/media/image84.jpeg)

Figure 79. Component support Plate

![A grey metal shelf on a box Description automatically generated](FYP_1/media/image85.jpeg)

Figure 80. Support for Tablet Holder

![A grey curved object on a table Description automatically generated](FYP_1/media/image86.jpeg)

Figure 81. Lower Torso

![A metal shelf on a box Description automatically generated](FYP_1/media/image87.jpeg)

Figure 82. Tablet Holder

![A person holding a metal piece Description automatically generated](FYP_1/media/image88.jpeg)

Figure 83. Head Support Plate/Top Plate

![A piece of paper on a table Description automatically generated](FYP_1/media/image89.jpeg)

Figure 84. Front Torso Parts

**3.12. Assembly:**

**3.12.1. Initial assembly:**

Removing the extra plate of peoplebot platform to start mounting the designed components was the first step. The mild steel plates had to be discarded at this point due to unforeseen issues and acrylic plates were manufactured. The assembly started by attaching the base plates, now of acrylic, to peoplebot platform to provide a stable platform for the subsequent assembly.

![](FYP_1/media/image90.jpeg)  
Figure 85. PeopleBot

According to design, the 3D printed parts were then assembled using keys, bolts, and screws to guide assembly and ensure all parts fit securely. A few adjustments were needed such as realignment of holes and tightening of all parts together to ensure proper alignment and functionality.

**3.12.2. Final Torso Assembly:**

According to design, the 3D printed parts were then assembled using keys, bolts and screws to guide assembly and ensure all parts fit securely. A few adjustments were needed such as realignment of holes and tightening of all parts together to ensure proper alignment and functionality.

![A small model of a table Description automatically generated](FYP_1/media/image91.jpeg)  
Figure 86. Arrangement of parts of torso

![A grey object in a room Description automatically generated](FYP_1/media/image92.jpeg)

Figure 87. Arrangement of Torso Before Mounting

![A grey podium with a screen and laptops Description automatically generated](FYP_1/media/image93.jpeg)

Figure 88. Torso Assembled

**3.12.3. Assembly Challenges and Solutions:**

  - **Material Substitution**

Challenge: At the beginning, the material used was mild steel plates but these were found to be ineffective and had to be removed due to some certain problems, namely misalignment and size variation.

Solution: They were replaced by acrylic plates which were manufactured earlier on in the production of the car. Acrylic is less challenging compared to the others, and if required, can be altered easily.

  - **Hole Alignment**

> Challenge: The Peoplebot plate could not be altered, leading to alignment issues after the acrylic plates manufacturing, with the mounting holes of the new plates and 3D printed components misaligned.

Solution: Re-drilling and aligning the holes were necessary to ensure a precise fit. Careful measurement and use of appropriate tools ensured successful re-alignment.

**<span class="underline">Chapter 4 - SYSTEM DESIGN AND IMPLEMENTATION</span>**

This chapter provides an overview of current approaches and tools concerning emotion and gender identification, speech recognition and web application. Different classifiers such as the CNN is also described.

**4.1. SYSTEM DESIGN AND IMPLEMENTATION**

**4.1.0 Dataset Preparation**

The emotion recognition model was trained using the FER-2013 dataset, a widely used benchmark for emotion detection containing over 35,000 facial images categorized into seven emotions: This study includes seven basic emotions; namely, Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. For the gender recognition, we used Adience dataset which includes collection of face images with labels as male or female. Model Architecture Picking the right model architecture or choosing our initial set of features is one of the most important steps in designing a deep learning model. These include:

**4.1.1. Emotion Recognition Model**

Model Name: Design of CNN for Emotion Recognition Architecture: Training: Optimizer: Adam Loss Function: Categorical CrossEntropy Validation Split: 20% Architecture: Input Layer: 48 \&times; 48 pixel images in black and white Convolutional Layers: The next four convolutional layers, with 32, 64, 128 and 256 filters respectively are created with kernel size 3×3 and ReLU nonlinearity. Pooling Layers: It should be noted that MaxPooling2D layers are included after each of the convolutional blocks. Flattening Layer: Changes the dimensionality of feature maps from 3D into the 1D feature vectors. Fully Connected Layers: The next is the dense layers with neurons of 128 and 64 and ReLU activation function. Output Layer: Another dense layer was added after which the SoftMax activation function was used to classify the eight emotions into seven.

**4.1.2. Training:**

Optimizer: Adam. Loss Function: Categorical Cross Entropy Epochs: 30 but 20 graph taken. Validation Split: 20%.

# ![A graph showing the difference between the loss and loss of the loss Description automatically generated](FYP_1/media/image94.jpeg)

Figure 89. Training Results

# **4.1.3 Model of Gender Recognition**

# Using deep learning for HARP, a pre-trained Convolutional Neural Network (CNN) model was used. This model was chosen for its good precision and reliability in gender identification based on facial images which eliminates the necessity of additional training. Built upon a CNN structure that achieves high accuracy as well as is relatively complex, moderate computational power is needed in the model.

# A typical trained gender classification model forecast was also checked against criteria pertinent to our system, such as accuracy. Thus, for the purpose of this project, the training and validation of the model was performed 100 times to check the regular fluctuations in loss and accuracy. The graph portrays the performance of the model revealing the training loss in which it is evident that the model is learning as shown by the reduction in the loss rate as learning progresses. However, considerable oscillations in the validation accuracy and loss mean the presence of overfitting. Although, training loss was less, the model did not generalize well to the validation data and hence there is a requirement of further tuning and may be use of state of art regularization techniques.

# With the help of this preprocessed gender-bearing model, it was possible to easily incorporate the gender determination into the sentiment analysis and thus enable the HARP robot to recognize the customers’ gender and tailor its responses depending on it. This matters in enriching the overall interaction experience and in integrating the gender of the user since this influences the robot’s behavior leading to more human like characteristics.

# Evaluation was done on the performance of the gender classification model which was then compiled with the trained emotion recognition model. These two models if integrated, help in the display of suitable emoticons on the running LCD of the robot hence improving the interaction of the robot to its human counterparts. It again demonstrates a highly modular and scalable concept of our solution where basic industry standard solutions can easily be integrated with specially developed services to provide a full solution.

![](FYP_1/media/image95.jpeg)

Figure 90. Training Loss Accuracy

![](FYP_1/media/image96.png)

Figure 91. CNN Confusion Matrix

![](FYP_1/media/image97.png)

Figure 92. Sequential Layers

![](FYP_1/media/image98.png)

Figure 93. precision Recall Flow

## **4.1.4 Real-Time Emotion and Gender Detection**

In execution of the system’s function to capture raw video information, the system uses a webcam to obtain a real-time video feed. Faces are detected using OpenCV’s Haar cascade classifier and then using the Tetris algorithm for sorting. The detected faces are normalized and resized next to meet the input constraints of the emotion and gender recognition algorithms. The result predictions are in real-time, and the respective emoticons are shown on a 7” LCD screen mounted on Jetson Nano.

**4.1.5. Displaying Emoticons on LCD:**

To get a more interactive response from users, there are specific emoticons for each gender. The gender of the predicted image is decided by the output of the gender recognition model while the emotion of the predicted image is decided by the output of the emotion recognition model. Depending on the gender, appropriate emoticons are displayed: Depending on the gender, appropriate emoticons are displayed and changed in real time.

Male and Female Emoticon Classes: The list includes happy, sad, neutral, angry, fear, surprised and disgust.

**4.1.6. Emoticons/Avatars:**

![](FYP_1/media/image99.png)

Figure 94. Surprised Male

![](FYP_1/media/image100.png)

Figure 95. Surprised Woman

![](FYP_1/media/image101.png)

Figure 96. Sad Male

![](FYP_1/media/image102.png)  
Figure 97. Sad Woman

![](FYP_1/media/image103.png)

Figure 98. Neutral Male

![](FYP_1/media/image104.png)

> Figure 99. Neutral Woman
> 
> ![](FYP_1/media/image105.png)
> 
> Figure 100. Disgusted Male
> 
> ![](FYP_1/media/image106.png)
> 
> Figure 101. Disgusted Woman
> 
> ![](FYP_1/media/image107.png)
> 
> Figure 102. Happy Male
> 
> ![](FYP_1/media/image108.png)
> 
> Figure 103. Happy Woman
> 
> ![](FYP_1/media/image109.png)
> 
> Figure 104. Fear Male
> 
> ![](FYP_1/media/image110.png)
> 
> Figure 105. Fear Woman
> 
> ![](FYP_1/media/image111.png)  
> Figure 106. Angry Male

![](FYP_1/media/image112.png)

> Figure 107. Angry Woman

**4.1.7. Raspberry Pi 4B**

Raspberry Pi4 Model B is an advanced feature-rich computer capable of handling several applications from simple IoT, media center to basic artificial intelligence and machine learning models. It is more used than other gaming platforms because of the cheap subscription service they offer, simple interface of their web page and forums, and the great response time of the community. One of the advantages of Raspberry Pi OS is they are easy to download and write on a micro SD card so the access to Raspberry Pi 4 can be easy.

![](FYP_1/media/image113.jpeg)

> Figure 108. Raspberry Pi 4B, 8Gb Ram

The Raspberry Pi 4 Model B is a reliable single-board computer perfect for computing and productivity, and for a multitude of functions at a low power requirement. It has various RAM versions (2GB, 4GB, and 8GB) and has an appropriate cost. The use of raspberry is complemented by the raspberry OS making it quite easy to set up and operational within a short span of time. The Raspberry Pi 4 performance is somewhat is fairly good for a Pi and with 8GB of RAM, it can take moderate amounts of AI work which makes it ideal for schools, home use, and even some industrial applications. The first thing one is required to do with it is to look at the Raspberry Pi 4 Setup Guide. Details of Raspberry Pi 4 Model B as indicated the technology parameters in the following table, Table 3.

Table 4. Raspberry Pi 4B, 8Gb Ram

| **Features**      | **Description**                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------- |
| CPU               | Quad-core ARM Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz                                       |
| GPU               | Broadcom VideoCore VI                                                                       |
| Memory            | 2GB, 4GB, or 8GB LPDDR4-3200 SDRAM                                                          |
| Storage           | microSD card slot                                                                           |
| Network           | Gigabit Ethernet                                                                            |
| Wireless          | 2.4 GHz and 5.0 GHz IEEE 802.11ac Wi-Fi, Bluetooth 5.0, BLE                                 |
| Ports             | 2 × USB 3.0, 2 × USB 2.0, 2 × micro-HDMI, USB-C for power, 3.5mm audio/composite video jack |
| GPIO              | 40-pin GPIO header, I2C, SPI, UART, PWM                                                     |
| Display Interface | 2 × micro-HDMI ports (up to 4Kp60 supported)                                                |
| Camera Interface  | 2-lane MIPI CSI camera port                                                                 |

**4.1.8. Functional overview:**

The latest Raspberry Pi 4 Model B has proven to be very useful in various computing tasks ranging from simple end-point devices to modest AI applications.

Core components of Raspberry Pi include:

  - **QuadCore ARM CPU**

> The Raspberry Pi 4 Model B uses a Broadcom BCM2711 System-on-Chip design manufactured by Raspberry non-profit foundation which is a chartered company and is powered by a 1. 5GHz quad-core ARM Cortex-A72 processor, this offers significantly better performance than the predecessor. It can operate multithreading and is good for everyday OS functions and moderate level artificial intelligence operations.

  - **Broadcom VideoCore VI GPU:**

> The Broadcom VideoCore VI GPU covers integrated functionality for high-definition videos and is sufficient for providing graphical intensity for simple graphics works and some gaming. It can also be used for parallel computing jobs that may include jobs within artificial intelligence and machine learning.

  - **8GB (LPDDR4) memory:**

> It is equipped with three variants depending on the RAM standard, where users will be able update depending on the performance preference required. The 8GB configuration is ideal where more memory is required for purposes such as machine learning and macro model simulations.

  - **Gigabit ethernet:**

> Thus, Ethernet connection of Gigabit and dual-band wireless LAN are available in Raspberry Pi 4. It also has Bluetooth 5. 0 for peripherals and other objects that a remote Bluetooth device can connect to.

  - **USB 3.0 Ports:**

> The incorporation of two USB 3. 0 ports is used for swift data transfer rate meant to enhance communication with other peripheral devices and storage devices.

  - **Micro HDMI Ports:**

> In this model, the Raspberry Pi 4 Model B comes with two micro-HDMI ports in which the board can drive dual 4K displays, which afford multimedia support and adjustable display options for various uses.

### **4.1.9. Operation**

The working of Pi is based upon the following steps: The working of Pi is based upon the following steps:

  - > Raspberry Pi 4 Model B can be interfaced to monitor, keyboard, mouse, and extra sensors/ actuators via GPIO ports. MicroSD as boot and internal storage, and mini-USB as the power supply.

  - > MicroSD as Boot and Internal Storage: Being a digital electronic device, Raspberry Pi 4 Model B too does not have an onboard storage and uses microSD card for booting and storage. An operating system image must be installed on the microSD card, for instance Raspberry Pi operating system.

  - > Speaking of interaction with the Object developed using the developer kit, there are two possibilities: Either have the kit together with the display, mouse and keyboard attached or in “headless mode” by connection from another PC.

Table 5. Comparison of initial setup using headless mode with display attached.

|                              | **Initial setup with display attached**            | **Initial setup in headless mode** |
| ---------------------------- | -------------------------------------------------- | ---------------------------------- |
| Monitor, Keyboard, and Mouse | Required                                           | Not Required                       |
| Extra Computer               | Not Required                                       | Required                           |
| Power Options                | Either Micro-USB or USB-C power supply can be used | USB-C power supply needed          |

#### 

**4.1.10. Libraries and Tools**

  - OpenCV: What is real-time use for image acquisition, processing, and face recognition? Specifically, OpenCV's cv2. VideoCapture function enables the web-camera control, while the Cascade Classifier function allows seeking for the faces in the video stream.

  - NumPy: Used for fast computation, data handling, storing image data and their respective labels in a NumPy array format and performing operations on arrays.

  - TensorFlow/Keras: Is often utilized as a framework for model construction and training of deep neural networks for deployment. To merge the models for emotion and gender recognition, the Keras Sequential API is employed.

  - Scikit-learn: First, the data is pre-processed and divided into subsets using this repo. For instance, the Label Encoder function is used for encoding categorical labels, while the train\_test\_split function splits the collected dataset into two sets of the training and testing data.

  - PIL (Python Imaging Library) and ImageTk: Tkinter needed for the window and image preprocessing and for displaying emoticons. PIL handles the opening and resizing of images, while ImageTk is responsible for showing these images in the Tkinter framework.

  - Tkinter: Develop graphical user interface for displaying using Tkinter for construction of windows and labels to give a graphical user interface for static and dynamic form of emoticons

**4.1.11 Technical Details of CNN:**

**4.1.11.1** **Convolutional Neural Networks (CNNs)**

  - Convolutions are the building blocks of a CNN and are used extensively in the architecture of the network. They are made of several learnable filters or kernels that are slid over the input image to yield feature maps. The filter should recognize a specific pattern in the input image and respond accordingly that is, edges, texture, or any other feature.

  - **Filters and Kernels:** Filters are square and smaller than the input data, and they move over the data. This is the size of the kernel and defines the size of the filter that will be used in the process. In our models, the size of the receptive field was set to 3×3, which is standard as it strikes a balance between performance and memorization of all sorts of details.

  - **ReLU Activation:** The Rectified Linear Unit (ReLU) activation makes a model non-linear by a certain degree hence making it capable of learning more complex patterns. Before, we learned that its write-off all the negative pixel values of the feature map and only retain the positive pixel values. It assists in reducing the vanishing gradient problem besides helping to speed up the training of the network.

**4.1.11.2** **Pooling Layers**

Pooling layers are incorporated to reduce the size of the feature maps and therefore, the size of the network as well as the number of computations. MaxPooling strides over the feature map with a patch that selects the maximum value from it, which through reduces the dimensionality of the feature map while maintaining optimal feature information.

**4.1.11.3** **Flattening and Fully Connected Layers**

Flattening changes from the 2D feature maps to the 1D feature vector, then it is input to the fully connected layers. These dense layers contain the maximum amounts of neurons which gives decisions relative to the features that are detected by the layers of convolution and pooling. The last layer in the models is utilized by applying SoftMax activation, to output the likelihood for the respective categories to finalize the classification tasks.

**4.2. Speech Synthesis**

In the evolving world of speech technology innovation has possibilities. Speech recognition technology enables spoken words to be turned into commands ushering in an era of hands-free communication and seamless interaction. On the hand speech synthesis brings text to life by creating speech that breaks through language barriers and improves accessibility. However, there are challenges, like dealing with background noise and ethical dilemmas that come with these advancements. Despite these obstacles the potential of these technologies to transform human computer interaction and enhance accessibility, in fields is unmatched. This promises a future where communication has no boundaries.

1.  **Need for Natural Human-Computer Interaction**

The rapid development of technology and its integration into human society have changed the traditional paradigms of human-computer interaction. The book, as a repository of the essence of the language, defeated time, becoming immemorial. This victory of mankind can be considered unquestionable. However, today, the secret life of signs has become known, becoming a partner to the interactive reality. Realizing this truth articulated in the humanization of the interaction of human and computer. Especially, realizing, the victory of the electronic computer was a temporary victory. It opened in the era of Nanotechnologies.

2.  **Role of Speech in Communication**

The rise of speech recognition technology embodies a new epoch of HCI, intensifying the user experience in terms of convenience and naturalness. This technology surpassed simply improving the performance and productivity of technical systems; it alters practices and enables innovation, strengthening the attitudes of people and amplifying society in terms of safety and accessibility. It has major implications in many branches of knowledge and there is a possibility of numerous beneficial applications. Since science remains and continuously develops, and speech recognition is a part of such a budding science, its potential only increases, while the potential is infinite and boundless.

3.  **Speech Synthesis and Recognition**
    
    1.  **Techniques used in Speech Synthesis and Recognition**

The program makes use of two libraries for speech synthesis and recognition: pyttsx3 and speech recognition. Pyttsx3 is used for text to speech (TTS) to make the software transform written letters to a language. On the other hand, the speech recognition library is used for speech to text (STT) through which the program can comprehend the user’s vocalization and write them in form of text. These libraries are useful for embedding voice components into an application and this enhances human computer interaction.

**4.2.3.2 Methodology**

**4.2.3.3 Overview of Speech Synthesis and Recognition**

The advancement of speech technologies has potential in the continuously changing environment. It is a speech recognition technology where words that are spoken can be translated into commands hence advancing to the next stage of having gadgets that respond to voice commands. Speech hand speech synthesis makes the text to speech where it gives speech a voice and helps to overcome language barriers and enhance language accessibility. However, there are some issues, for example handling of noises in the background and some ethical issues which are associated with the recent innovations. However, these inventions are yet to be fully realized they have the potential of revolutionizing how man interacts with the computer with particular reference to accessibility in specific fields. This brings a future whereby there will be no limitation to the communication that will be made.

4.  **Implementation Details**

**Speech Synthesis:**

1.  **Initialization:** The text-to-speech engine is started through engine = p.init().

2.  **Property Configuration:** In order to define the speech rate, it is decided that the speed should be 120 words per minute by using engine. setProperty('rate', 120). The voice is selected from among all the available ones with voices = engine. getProperty('voices') and engine. setProperty('voice', voices\[1\]. id).

3.  **Text-to-Speech Conversion:** The speak(text) function has as input a string and outputs a string of speech using engine. say(text) followed by engine. to make it read and generate the spoken output; the runAndWait() is used to process.

> ![](FYP_1/media/image114.png)
> 
> Figure 109. Snippet for TTS
> 
> **Speech Recognition:**

1.  **Recognizer Setup:** This is what is returned when an instance of the Recognizer is created with r = sr. Recognizer().

2.  **Audio Capture:** Using the microphone as the audio source, the energy threshold for noise filtering is set with r.energy\_threshold = 10000, and ambient noise adjustments are made using r.adjust\_for\_ambient\_noise(source, 1.2). The audio input is captured with audio = r.listen(source).

3.  **Speech-to-Text Conversion:** The takeCommand() function processes the captured audio, attempting to recognize and convert it to text using query = r.recognize\_google(audio, language="en"). If successful, it returns the interpreted text; otherwise, it returns an error message.

> ![](FYP_1/media/image115.png)
> 
> Figure 110. Snippet for STT

4.  **Accuracy and Error Handling**

The takeCommand () function is designed to haul speech recognition with a focus on accuracy and robustness. Here be how it take these aspects into account,

1.  **Accuracy Enhancement:**

> The function utilizes r.adjust\_for\_ambient\_noise (source, 1.2) for adapting to ambient noise, inhaling the recognition process by filtering out irrelevant sound. This ensures the audio getting be as clear as feasible. Energic Threshold Positioning: By setting.energy\_threshold = 10000, the recognizer be fine-tuned to be more touching to the user's voice while ignoring quieter background noises.

2.  **Error Handling:**

> The try block attempt to recognize speech using Google's speech recognition service (query = r.recognize\_google(audio, languish="en")). If the recognizer cannot comprehend the audio (sr.UnknownValueError), the user is asked to select their input with a spoken message ("I hadn't catch that. Would you repeat?"). The function then momentously calls itself (return takeCommand()) to let the user to reattempt. Service Errors: If there is a mishap with the recognition service itself (sr.RequestError), the function informs the user of the problem ("There be an issue with the service; please reattempt later.") and returns None. This ensures the user is conscious of the issue without leading the program to crash or behave unrecognizable.

3.  **Python Chatbot Development: A simple Starting Point**

Created a basic Python Chabot application that assists people with everyday tasks which might include watching videos in the YouTube or get information from the Wikipedia. It was beneficial to gain an understanding of how such technologies work; particularly recognizing the commands of a chatbot and performing them as fundamental components in creating more complex chatbots.

It asks the user a question ‘hello’ and then it moves to the command mode in which it waits for the commands to then start an infinite loop. The core utilizes a different approach that is aimed at identifying specific sets of commands, as well as the corresponding actions to be executed. For instance, if the user enters a command in the chatbot like: Open YouTube, the chatbot interprets it and then uses the webbrowser module to open the Google’s YouTube home page. This process requires analyzing the recognized text to correlate the text identifiers with the predefined actions.

The command identification process is structured as follows: The command identification process is structured as follows:

**Command Interpretation:** The chatbot waits for input from the user and interprets the entered output as a command.

**Keyword Matching:** It can then match the recognized text to a list of standard phrases and voices commands, such as, ‘open YouTube’ or ‘open Wikipedia.’

**Action Performance:** For the matched command, the chatbot performs the predefined action based on the action set that like visiting a certain website using webbrowser.

For instance, the primary conversational features include such phrases as ‘Open YouTube,’ ‘Open Wikipedia,’ and ‘Open Google’ and the chatbot opens the corresponding sites.

Further, it responds to requests to stop or resume having a conversation, making it easy to manage the interactions as well. This project has been instrumental in understanding how chatbots can: This project has been instrumental in understanding how chatbots can:

**Identify and Execute Commands:** Learning and translating particular scenarios, it is clear that the main principle of using this chatbot is the command-based approach.

**Enhance User Interaction:** Such immediate reactions to commands like opening said website make the assistant seem active and thus enhances the feel of a good smooth user experience.

**Error Handling and Flexibility**: Simplifying error handling means that the chatbot will be able to respond to unrecognized commands and posted messages politely and asking the user to try again or the chatbot will inform their side that there is a problem.

These aspects are something that have helped me prepare to build other and better chatbots. With the help of mastering of the command recognition and the execution I can now include more complicated operations and increase the rate of precision and offer the users a higher and wider level of interactivity. This systems’ foundational knowledge aids in preparing the ground for the future incorporation of other detailed characteristics like natural language understanding, contextual perception, and improved speech features**.**![A diagram of a flowchart Description automatically generated](FYP_1/media/image116.png)

Figure 111. Python Chatbot Development: A simple Starting Point

4.  **Capabilities and limitations of Gemini and Open AI**

> Within the scope of the Retrieval-Augmented Generation (RAG) pipeline, both Gemini Pro and OpenAI’s GPT-3. 5 provide different opportunities and challenges. Due to its efficiency based on generative algorithms, Gemini Pro offers excellent query management without any additional costs, which makes it suitable for deploying in large enterprises with strict budget requirements. Its disadvantages are as follows: The use of this website is limited by rate limits: 15 requests per minute and 1,500 requests per day. Conversely, OpenAI's GPT-3. 5, which also comes with comparable usability and result quality, has high costs and charges depending on the usage of tokens, which may lead to higher expenses if used frequently. Unfortunately, it lacks rigorously enforced rate limits, which is beneficial in cases where a high querying rate is essential. Both models are compatible with the RAG framework providing effective support for the process of retrieval and generation, however, the choice between them might be made in terms of cost, rate limits or specific requirements to the application.
> 
> ![](FYP_1/media/image117.png)
> 
> Figure 112. Aspects

5.  **Implementation of Gemini Chatbot**

> The plan of the Gemini chatbot requires several stages, using Google’s Generative AI (Gemini) model and speech recognition/synthesis in order to develop a voice-driven assistant. Below is a detailed explanation of the implementation process:

**1. Importing Necessary Libraries:**

To begin with, the required libraries are imported:

**Google Generative AI (Gemini):** It is used in this library to generate images from textual descriptions without directly querying Google services.

**Speech Recognition:** This library also provides the functionality of Voice Recognition by recognizing voice commands given by the user.

**Pyttsx3:** This is used for speaking the text responses provided by the chatbot as a speech that can be understood by the humans.

1.  **Configuring the Generative Model**

> The generative model is configured with specific parameters to control the nature of the responses:
> 
> **Temperature:** Determines the degree of unpredictability in the solutions returned by the algorithms. A value of 0.6 helps maintain a proper degree of creativity and cohesiveness.
> 
> **Top-k:** These parameters calculate the number as well as diversity of the output by taking into account the first probability of values most likely to occur.
> 
> **Max Output Tokens:** Sizes the output in terms of the total number of tokens allowed within the generated response.

Further, privacy settings are established to block out negative content which includes harassment, hate speech, pornography, and disastrous content. These configurations and settings are fed to the model and then the API key is given for authorization**.**

**3. Starting TTS engine**

> For the text-to-speech preparation, pyttsx3 library is adopted and it also initialized. These settings consist of speech rate and voice properties so as to provide good clearances of audio responses. This makes it possible for the chatbot to give verbal responses to the presenter.

**4. Capturing Voice Commands**

> This section of the simulator has been developed to capture any voice commands that the user may have to give. In order to input the user’s voice, the speech recognition module is used where the microphone is connected to record the user’s voice. The captured audio is then translated into textual feedback using Google Translate voice-to-text tool. For any error that occurs during this process an associated error message is produced. O SIMD\_MRI can occur at any of the different stages of its utilization.
> 
> **5.Generating Responses**
> 
> Function is defined for constructing appropriate response using the Gemini model. The verbally provided input translated into text is fed to the generative model which then analyzes the input and formulates an appropriate response. It is then translated back to text; in case it was in language code format.
> 
> **6. Main Loop for Enduring Interaction**
> 
> The last while cycle makes sure the interaction with the user never stops, moving through the different stages of the program. When the program is activated, the chatbot says hello to the user and enter a loop that is list for commands from the user and generates a response that in an audio format to the user. The conversation loop continues investing in the result until the user wants to stop interacting with the tool by typing something like “exit.”

![A diagram of a model Description automatically generated](FYP_1/media/image118.png)

Figure 113. Implementation of Gemini Chatbot

3.  **Introduction to RAG – Retrieval Augmented Generation**

RAG is an abbreviation for Retrieval augmented generation, a system utilized to assist LLMs model such as gpt-4, Gemini, Gemma, LLAMA2, MISTRAL assist the model to provide the correct response by using facts and information from other sources in order to decrease the probability that the LLMs will lead to wrong information. It is another approach in language model generation that takes extra context into consideration. This is often achieved by accessing the required text from a big pool of documents then using the extracted information to generate the text. With RAG, the LLM is able to utilize knowledge and info that is not necessarily maintained in the weights, which means that it can be outside the training space.

LLMs have:

**1**. Limited knowledge access.

**2.** Lack of transparency: The clarity and relevance of information remain questionable, and the answers produced by this artificial intelligence tool often fail to meet users’ expectations and serve as sufficient explanations to legal questions.

**3.** Hallucinations in answers.

4.  **RAG Architecture**

In RAG we want to connect the LLMs with other information (in the form of document that data is store in database and we will connect it with LLM). If we want to make our model much more efficient in that case we will have to use RAG, to make our model task specific or domain specific.

1.  **Ingestion**

> Data can be in any format in our case it is in the form of pdf and then we will extract the data after extracting we are going to divide it into various chunks. Then after it we are going to convert it into an embedding. Embeddings are nothing just the numerical representation of the data. Then we will build semantic indexes then it is store in vector database. This whole part comes under ingestion.

**2. Retrieval**

> Obtaining knowledge from the vector database. Now whenever the user asks any query, first we convert it into embedding then after performing semantic search and based on that it will give top k responses (ranked results).

**3. Generation**

> Now after retrieving the results, it gives the top ranked responses to LLM along with the query then after doing similarity search the answer is generated in this part and user query is solved.

**4.2.9 Integrating RAG and Gemini Chatbot**

**4.2.9.1 Design Considerations for Combining Systems**

> The approach of integrating Retrieval-Augmented Generation (RAG) with the Gemini chatbot has certain limitations which must be addressed in relation to the functionality and efficiency of the final product. Here are the key considerations:

**1. System Architecture:**

  - **Modular Design:** The architecture should be highly de modularized, that is each part in the architecture should be capable of working separately from others which include RAG system, Gemini chatbot, text splitter, vector database, etc. Key things about this modularity include: The work is divided into smaller manageable parts when developing, debugging and even when trying to incorporate new changes in the future.

  - **API Integration:** It is also needed to develop an understanding of how the system will be interacting with the external APIs, e. g., the Gemini API, so that communication would be properly secured and optimized as much as possible. These should be properly handled in the correct process because these are vital to the management of the APIs.

**2. Data Management:**

  - **Document Handling:** This is something that one needs to ensure that they get right all the way through the handling and processing of the documents. Loading of documents involves organizing documents so that they can be accessed easily, as well as fragmenting documents for further optimization and making them retrievable.

  - **Context Management:** The ability to keep context over a series of user interactions is arguably as important to enable a meaningful and contextually appropriate response. The system should be capable of storing and recalling context according to any schedule.

**3. Performance and Scalability:**

  - **Efficient Retrieval:** Search for chunks of documents must be easily facilitated by the system so as to enable retrieval in a short span of time according to the users’ query. This is a prerequisite of a vector database that is well indexed and the manner in which the retriever is designed.

  - **Load Handling:** In designing the application, it should be able to accommodate multiple user interactions while at the same time not compromising on performance. This can entail rotation and initialization of load as part of the regular activities of the system.

**4. Response Generation:**

  - **Prompt Engineering:** This means that the development of proper prompts is major for helping the language model navigate and give the proper answers. The prompts must be readable, specific, and sober and should ideally be created for a certain purpose and use.

  - **Fallback Mechanism:** There should be a fallback to the system so that in case of inadequate response to the issue raised in the question, the RAG system should respond that the model Gemini will be used to provide appropriate answer.

**5. Safety and Reliability:**

  - **Safety Settings:** It is important to set appropriate security measures regarding the generative model, to prevent it from creating unwanted or obscene content. These should be set to either block or filter out the selected categories to cater for sensitivity.

  - **Error Handling:** It is also necessary to have effective error handling procedures in the application to handle cases such as failed API requests or handling of inputs not recognizable by the program.

**6. User Experience:**

  - **Natural Interaction:** The chatbot should be human-like and friendly in the way that it responds to he or she who is conversing with it. It also promotes the level of user interaction and satisfaction.

  - **Clear Instructions:** Giving directions and feedback to the users on whether they have typed correctly, or the system has failed to understand them, should be given to enhance the functionality and user experience.

Thus, upon incorporating RAG and the Gemini chatbot, these design aspects can form an effective and effortless conversational assistant. This combination uses the best of both systems to ensure that the questions posed yield correct, contextually appropriate answers and at the same time delivers high performance, and stability.

**4.2.9.2 Implementation Details of the Integrated Chatbot**

The process of deployment of the integrated chatbot requires structures, which work harmoniously within the system, as follows. Here are the steps involved:

**1. Initialization:**

  - **Loading Models and Embeddings:** In the beginning, a ChatGoogleGenerativeAI model should be created with the Gemini Pro API installed; prepare GoogleGenerativeAIEmbeddings for document embedding. These components as described comprise the core of the chatbot’s response formulation.

  - **Document Loader and Text Splitter:** Avoiding manual pre-processing of the text, we can load documents using the PyPDFLoader and split the text into easily manageable portions using the CharacterTextSplitter. This makes it possible to process large documents as well as to view one document within another document window.

**2. Vector Database and Retrieval:**

  - **Creating the Vector Database:** It will be useful to organize all document chunks in the vector form using Chroma for converting them to a vector database. This ensures ease in accessing the necessary sections of the documents from the different fields based on the queries done by the users.

  - **Setting Up the Retriever:** Specifically, the retriever should be configured to extract appropriate chunks from the vector database in order to make the utmost information accessible in cases of response generation.

**3. Prompt and Response Chains:**

  - **Prompt Template:** Specify a template to prompt the model on the right conversation language and aptitude to respond to. This makes it easy to remove ambiguities of the given inputs and keep the responses natural, relevant and specific.

  - **Combining Documents Chain:** Round up the relevant documents of the identified prompt and form a sequence of document snippets that can be strung together by the language model to produce text that is contextually relevant and coherent.

**4. Main Execution Loop:**

  - **Listening for Queries:** Addition to this, it should be able to act increasingly in a manner where it is constantly listening to queries from the user and responding to them in a real-time manner. This includes identifying results specific keywords to perform actions such as exiting or to input date and time information.

  - **Response Generation:** Employ the retrieval chain to formulate responses according to the user input string. If the RAG system is not able to offer an appropriate answer, slip down to the use of the Gemini model to make sure that the user is provided with the most suitable answer.

  - **Reinitialization:** It is wise to periodically reinitialize components to ensure they tap efficiency in achieving user new interactions.

**5. Error Handling:**

  - **Robust Error Handling:** The speech part also needs to have measures for handling error conditions such as when the model fails to recognize the speech or when there is an error with the API request. This helps make the use of the systems fast and support a faster flow of operation throughout the organization.

With these implementation details, the integrated chatbot would be able to refer and respond to the messages appropriately as well as sustaining high performance and reliability. This setup shows the possible way to implement the integration of RAG and the Gemini chatbot, which will open the doors to further developments in more sophisticated conversational AI.

![A diagram of a computer Description automatically generated](FYP_1/media/image119.png)

Figure 114. Final Integrated Chatbot with RAG and Gemini Chatbot

**4.2.10 Results and Evaluation**

**4.2.10.1 Evaluation Metrics**

The evaluation of the integrated chatbot system encompasses several critical metrics to gauge its performance and effectiveness comprehensively:

**Accuracy:** The extent to which the answers given by the chatbot are appropriate for the material presented in the context to which as much relevancy as possible is provided, and the information contained in the material is reliable. By keeping everything in mind the accuracy of RAG is about 80% to 90%. So, by combining both the RAG and Gemini Chatbot its accuracy gets much better.

**Response Time:** The amount of time that a chatbot utilizes to respond to a question posed by the customer and is quite relevant for scenarios where content is expected to be shared within a given duration. It takes few seconds to answer a question.

**Fluency:** How humans like the chatbot and how logical its answers the results show that the chatbot is trained to reliably respond in a certain manner. But the response is based on the data we provided to the pdf document from where the system is fetching the answer,

**Relevance**: Notes the extent of resemblance of the responses it provides with the user’s query, meaning that the answers given should match the questions asked.

These are the metrics that combined give a good overall picture and the potential and limitations of the chatbot.

**4.2.10.2. Comparison of Response Quality and Fluency**

Relative to response quality and fluency, there was no significant difference observed between the three groups, indicating equal proficiency in the language skills applied.The comparison of response quality and fluency between the Retrieval-Augmented Generation (RAG) system and the Gemini chatbot revealed significant insights:

  - **RAG System:**

<!-- end list -->

  - **Strengths:** Assesses high because the created model incorporates contextual details to try searching and narrow the information and then providing answers together with support documentation.

  - **Weaknesses:** This is true except where the two writing chunks are synthesizing; this creates hurdles in fluency and coherence.

<!-- end list -->

  - **Gemini Chatbot:**

<!-- end list -->

  - **Strengths:** Developed in speaking and reacting quality; has highly developed generative elements so that conversational and discursive fluency and feel are possible and natural.

  - **Weaknesses:** However, in the context of contextual relevance it is less of an accurate presenter at times thus the accuracy of the response in a context may be slight.

That is, when these two systems are integrated, they capitalize on both: giving precisely positive answers which are appropriately appropriate and spontaneous.

**4.2.10.3 Analysis of Efficiency and Resource Usage**

Efficiency and resource usage are crucial factors for the practical deployment of the chatbot:

  - **Memory Usage:** Documents loader, text splitter, and vector databases take significantly large space and most specifically are used for large documents; therefore, there should be better memory management.

  - **Processing Time:** This means the response time is dependent on the complexity of the query and the number of document chunks resulting from the search which may have an impact on the user.

  - **API Calls:** Charge of the Gemini service is a repetitive process that poses time response implications and should be done effectively to enhance performance.

These are the facets that should be optimized to improve how the chatbot is going to work with little or no hindrance as well as in controlling costs.

**4.2.10.4 Effectiveness of Combining RAG with Large Language Models**

RAG with the Large Language Models is able to leverage knowledge and information that is not necessarily in its weights, providing it access to external knowledge base.

  - RAG doesn’t require model retraining, that saves time and computational resources.

  - Improved relevance and accuracy.

  - Handling open0domain queries.

  - Reduced generation losses.

  - Human-AI collaboration.

**4.2.10.5 Interactive Chatbot Performance**

To assess the usefulness and efficiency of the Bot the performance based on its ability to solve different types of problems was tested especially the ones which are more complicated.

**4.2.10.6 Effectiveness in Handling Complex Queries**

The chatbot demonstrated proficiency in managing complex queries that involved multiple pieces of information:

  - **Contextual Understanding:** Through the integration of RAG, it was possible for the system to capture and assemble specific sections of documents that are closely related and pertinent to the questions posed and deliver precise and contextually appropriate answers.

  - **Generative Responses:** The Gemini chatbot was effective in the sense that the responses were always as smooth as possible and well-articulated in every way from one message to the next.

  - **Fallback Mechanism:** When the RAG system was inadequate in furnishing certain information, the Gemini chatbot supplied adequate generative answers that maintained the interpersonal flow of the conversation.

> Altogether, the integrated system was able to handle interactions of different levels of complexity with a high degree of success in that it provided sufficiently accurate and easy to use interactive responses.

**4.2.11 Limitations and Future Work**

Nonetheless, some challenges were observed during the chatbot development and implementation.

**4.2.11.1 Challenges Encountered During Development**

  - **Integration Complexity:** Anticipating and responding to context and enabling transitioning between responses were particularly challenging because of the integration of RAG and generative methods.

  - **Resource Management:** Managing memory and processor also turned out to be an area of concern when attempting to do some computations especially where large sets of data were involved.

  - **Error Handling:** Having to adopt efficient ways of error controlling in situations where errors are produced when the program is making its API calls or when the program gets inputs it cannot decipher.

These complications involved successive designing and the practice of perpetual trial in designing to be able to address the challenges that arose.

**4.2.11.2 Potential Areas for Improvement and Further Research**

Future work can focus on several areas to enhance the chatbot's performance and capabilities:

  - **Multilingual Support:** At the moment, it is just in English as mentioned earlier, while the availability of other languages ​​will be part of the further development of the chatbot. The former is one aspect which can be expanded to include other languages such as Urdu and will prove beneficial in this regard. This requires the use of the higher-quality multilingual Natural Processing Language tools.

  - **Improved Speech Recognition for Urdu Names:** Because of which it is not very much effective in the system for identifying names in Urdu language accurately. Future work on other pin-pointed useful APIs that can be applied on STT in aspects focusing on Urdu will be helpful.

  - **Advanced Context Management:** That is why the diversification of approaches to tracking and preserving the context appears to me as highly beneficial to make the response more relevant and continue the interaction in the same non-intrusive manner.

  - **Scalability:** Therefore, increasing the capability of the system for handling even larger amounts of data add the same time used by other users brings the aspect of scalability and more stability.

  - **User Feedback Integration:** Such features can aid in achieving the best response for every individual that posts their inquiry and improve the experience of customers who use the site.

  - **Enhanced Safety Measures:** A similar concern applies to the prevention of the generation of unwanted products: it is also necessary to enhance barriers restraining the publication of proposals as well as the control of processes leading to the generation of unwanted products.

These are some of the areas that the chatbot can be further improved to not only provide higher accuracy in client-based answers but also the improved interaction flow.

2.  **User Graphical Interface**

**4.3.1 Web application Technologies for Users**

The web application is the central interface between the user and HARP and is perhaps the application used most often by the users. The GUI is developed to be displayed on a tablet that is installed on the robot to allow the users to easily interact with the system. Due to the complexity of the design of the current version and existing programming, the development process required such stages as the creation of a convenient and adaptive interface using HTML, CSS, and JavaScript.

1.  **HTML Structure**

The HTML structure defines the layout of the web application, including various sections for different functionalities: The HTML structure defines the layout of the web application, including various sections for different functionalities:

2.  **Contact Us**

This section contains a form and a contact list that can be used to enter personal data and create a message.

**4.3.1.3 Direction & Availability**

It also contains customer service details and a directory that will guide the customer to the places of interest.

**4.3.1.4 Appointments**

This section enables users to schedule appointments through a voluminous form thereby providing detailed information.

**4.3.2 CSS Styling**

CSS is used for presenting the appearance of the web application and maintaining its design on screens of different sizes. This involves modifying traditional aspects such as format, palette and using techniques such as Flexbox and Grid to enhance usability.

**4.3.3 JavaScript Interactivity**

JavaScript is used for enhancing the web application and providing the logic for it. This is used for form submissions and voice commands, as well as API communication with the backend. Key functionalities implemented using JavaScript:

**4.3.4 Speech Recognition:**

> Recording the user speech input and then transcribing it down in terms of language processing.

  - **Form Validation:**

> Validating some of the inputs from the user before submitting and processing them.

  - . **API Integration**

> Transferring data from the Flask back end.

In this context, the Flask framework is used to enable API interface. Flask is the main back end for this Web interface it is solely in charge of the data processing and communication between the Web front end and other back end services. Flask is web development framework in python.

**4.3.5. API Endpoints**:

The Flask application includes several endpoints to handle different types of requests:

**/audio:** Receives basically, audio or JSON data which contains transcriptions from the web application. It analyzes the information and provides the result to the Free version.  
  
**/appointment:** Included to manage appointments from the online booking form where it receives the data from the form and write the data to a file.

**/contact**: Handles the contact information received from a user and stores this information for use by the system at a later time.

**4.3.6. Integrating LangChain and Google Generative AI**:

The application can also incorporate LangChain and Generative AI models from Google for backend processing improvements. These tools make natural language input recognition and response generation possible, by assisting the robot in the interpretation and formulation of human queries. The integration involves:

  - **LangChain**:

> This is the type of library that offers language model development tools and an environment to work with text data.

  - **Google Generative AI**:

> Question-answer models that give answers in relation to the likely contextual interrogations and queries submitted by the user.

  - **Speech Synthesis and Recognition:**

> The speech synthesis and recognition module enables the web application to engage in interaction with clients using voice capabilities. It requires the ability to recognize the user’s voice and transcribe it into text as well as analyze the text and respond in kind, orally.

  - **Speech Recognition:**

> Currently, the web application deploys the Web Speech API for speech-to-text recognition of the user’s utterances. This includes initializing the desired type of speech recognition instance, recording an audio source, and managing the transcription phase.
> 
> In this case, the data will be sent to the Flask backend. On the next step, after the speech is transcribed, the obtained text is sent to the Flask’s backend using API. This is achieved by using an XMLHttpRequest (XHR) for sending the data to the correct endpoint thru POST.

  - **Backend Processing:**

> The text transcribed from the Flask backend is interpreted by the LangChain as well as Generative AI models by Google. The response further passes through a process of formatting where it is transformed to JSON format before being sent back to the web application.
> 
> **Displaying Response**:

  - On the web application side, it gets the JSON response, parses the response to get the data required and display to the user. Furthermore, the text response is again converted in the Web Speech API and speech recognition into speech form so the user will be able to hear the literal response.

**4.3.7. Implementation Details**  
**4.3.7.1 Frontend (HTML, CSS, JavaScript)**

The frontend of the web application uses the standard technologies applied to Internet web sites. Key components include:

  - **HTML**: Specifies how the organization of the application, contains the section for the contact information with the location navigation and appointments.

  - **CSS**: This process involves giving the application colors, fonts, and layout that make it aesthetically appealing to the users and also appropriately designed to rotate between different devices.

  - **JavaScript**: Controls the behavior of the interface by accepting, for example, voice input and transmitting data to the backend, as well as displaying the content received from the server on the page.

**4.3.7.2 Backend (Flask, Py-Term, OpenAI Text Generation, Google Generative AI)**

The backend of our system is implemented as a web application using Flask, a Python lightweight web framework. Flask Serves as an intermediary between clients and the server handling incoming requests while coordinating endpoint routing.

**4.3.7.3 Code Integration and Workflow  
**

As for API, it connects and synchronizes the frontend and backend elements. It should be noted that in the case of interaction between the user and the web application, the displayed data is transferred to the backend to process information. The received data gets analyzed at the backend layer and a response is generated and passed back to the frontend for display to the user. This overall continuity makes the user interface smooth while also enabling interaction.  
  
**Example Workflow**

  - User Interaction: The user interacts with the devices by talking towards it, for instance, asking a question in the devices or proffering an answer to a question posed by the tablet.

  - **Speech to Text:**

The speech to text conversion process is done in the web application section using Web Speech API where it records the speech input and converts it to text.

  - **API Call:**

Once transcribed, the text is transmitted to the Flask backend through an API call and is then stored in the appropriate database.

  - **Backend Processing:**

The application analyzes the input in the Flask backend, utilizing both LangChain and Google Generative AI, to provide a response.

  - **Response Handling:**

The response is send back to the web application where it is presented to the user as text and converted to voice for further audible feedback.

This can be achieved through an appointment booking and contact management web application based on the characteristics of patient data.

Such features as appointment scheduling or contact list offer a wide choice of options to the users. Customers can simply input relevant details into forms in order to make an appointment or provide contact data, and these forms are then passed to the back-end.  
  
**4.3.8 Process Overview**

**4.3.8.1 Form Submission**

One of the common features is entering or providing data through forms and submitting them via the web application.

**4.3.8.2 Data Extraction**

The input stimuli are received and processed to a Flask backend from where then it send back to frontend.

**4.3.9 Conclusion**

The chosen approach has been successfully illustrated on a real-life sentiment and gender recognition application based on deep learning models. The combination of these models with Raspberry Pi 4B and the demonstration of emoticons on the LCD screen showed that future human-computer interfaces have a lot of opportunities. The future work could also consider enlarging the list of embedded emotions, as well as connecting it with other Internet of Things for more engaging functions.

**4.3.10 Code Snippets**

**4.3.10.1 Code A**

![](FYP_1/media/image120.jpeg)

**<span class="underline">  
</span>**Figure 115. Gender and Emotions 1

![A screen shot of a computer program Description automatically generated](FYP_1/media/image121.jpeg)

**  
**Figure 116. Gender and Emotions 2

![A screen shot of a computer Description automatically generated](FYP_1/media/image122.jpeg)

**  
**Figure 117. Gender and Emotions 3

**4.3.10.2 Code B**

![A screen shot of a computer program Description automatically generated](FYP_1/media/image123.jpeg)

**  
**Figure 118. Main 1

![A screenshot of a computer program Description automatically generated](FYP_1/media/image124.jpeg)**  
**Figure 119. Main 2

![A screen shot of a computer program Description automatically generated](FYP_1/media/image125.jpeg)**  
**Figure 120. Main 3

**4.3.10.3 Code C**

![](FYP_1/media/image126.png)

**  
**Figure 121. Web1

![A screen shot of a computer program Description automatically generated](FYP_1/media/image127.png)**  
**Figure 122. Web2

![](FYP_1/media/image128.png)

**  
**Figure 123. Web3

**<span class="underline">Chapter 5 - CONCLUSION AND FUTURE WORK</span>**

**5.1 Significance of Humanoid Assistive Robotic Plateform**

A new concept of the Humanoid Assistive Robotic Platform (HARP) is one of the most significant steps toward the adaptation of robotics in human-filled surroundings. The use of complex modules and great technologies in HARP makes it provide regular and even ceaseless support in numerous tasks, as well as increasing productiveness and quality of life among the affected person/s. Its facility to train and drive around without human help, recognize emotions, and respond are a vast advancement towards having robots that can freely interact with humans. As demonstrated, HARP has the potential to bring tremendous changes to several economic sectors; especially to those in which close interaction between human and robots can be applied – healthcare, manufacturing, customer services, etc. In addition, the modularity of this intervention enables its enhancement and insertion in multiple contexts and it can therefore be postulated to have significant implications for society. In the end, HARP is a very beautiful example of using contemporary technologies in combination with the evaluation of person needs at the present stage of technological advancement and creating robots that can be the next stage of human-robot symbiosis that would enhance our environment and make it more effective and friendly.

**5.2 Conclusion**

Humanoid Assistive Robotic Platform is a new generation assistive robot, showcasing many features that should make it pretty useful in that area of application. The core of the current study is developed with the PeopleBot framework added to several other features that create HARP and render a complex and stimulating environment. Some of the sub activities of this project are generation of a CAD model to produce a printed on a 3D printer and assembly on the PeopleBot making it strong and eye-catching in structure. The proposed project is an emotional detection and display module, which is developed on PyTorch, for Raspberry Pi, for reading and reflecting human emotions for better interactive communication between the human and the robot. The integration of a speech synthesis module using the RAG model, the coupling of tools such as Gemini, and the cooperation of ChatGPT enhance the functionality of the robot in terms of communication, particularly in the context of ecological and dialogically sensible interactions. The HTML, CSS, and JavaScript code creating a webpage are integrated with Flask to work as a front-end view to control the actions of the robot.

**5.3 Future Work**

There are areas where future work can lead to enhancement and improved operation of HARP; The areas that are most in need of development is the possibility of allowing it to navigate on its own. This includes increasing the object recognition and proactive search algorithms to enable HARP to walk and perform the search function on its own. Another area of focus is mechanical improvement that includes swapping the fixed neck and head with a gimbal to enable the dynamic head motion. Moreover, including robotics that contain arms, motors will enhance the impression and effectiveness of HARP like human arm. The project of the current model being implemented on an web camera in the realm of emotion detection and Sentiment Analysis will increase the level of accuracy. Increasing equational accuracy and further porting this model for Jetson Nano will improve the performance and integration. For the speech synthesis and chunking necessary for speech recognition, there is a need to add more language support to the chatbot; Basic Urdu Language Support through advanced Natural Language Processing tools must be incorporated. This is also important since the inclusion of more precise APIs can enhance the abilities of the software to better identify and train the Urdu names for getting a better replication by Speech Recognition. Enhanced context management will improve the applicability and topicality of responses; the scalability of the system will let one work more efficiently at processing the amounts of data and, simultaneously, users. The inclusion of user input features will constantly enhance response precision and quality, coupled with the use experience. The increase in content creation and sharing platforms demands improved security barriers to eliminate the creation and sharing of unwanted or even dangerous content.

The main foreseeable prospects include the further development of the above-listed web application and backend. Bearing these facts in mind, refining the UI of web application and incorporating additional features is likely to improve user experience. Optimizing the backend server is important when integrating for more user subscriptions and richer interaction with the server. Strengthening security protocols would help safe-guard users data and prevent other unauthorized parties from gaining access. Moreover, creating and implementing a program to schedule and coordinate the appointments on the web-base platform shall allow users to make appointments effectively.

**<span class="underline">REFERENCES</span>**

\[1\]. R. Karunasena, P. Sandarenu, M. Pinto, A. Athukorala, R. Rodrigo and P. Jayasekara, "DEVI: Open-source Human-Robot Interface for Interactive Receptionist Systems," 2019 IEEE 4th International Conference on Advanced Robotics and Mechatronics (ICARM), Toyonaka, Japan, 2019, pp. 378-383, doi: 10.1109/ICARM.2019.8834299.

\[2\] Mankar, M., Sayyed, A., Hedau, N., Fating, G., & Mahajan, M. (2020). Taking Receptionist Robot. *International Research Journal of Engineering and Technology*, *7*(2), 1265-1267.

\[3\] Rajput, H., Sawant, K., Shetty, D., Shukla, P., & Chougule, A. (2018). Implementation of voice based home automation system using Raspberry Pi. *International Research Journal of Engineering and Technology*, *5*(5), 2771-2776.

\[4\] Manoharan, P. A., Vivian, J. M., Winson, N., Azariah, S., & Kousalya, G. (2018). DESIGN MODELING AND FABRICATION OF HUMAN-HUMANOID ROBOT COMMUNICATION. *Pakistan Journal of Biotechnology*, *15*(Special Issue-II), 1-7.

\[5\] L. Ismail *et al*., "Face detection technique of Humanoid Robot NAO for application in robotic assistive therapy," *2011 IEEE International Conference on Control System, Computing and Engineering*, Penang, Malaysia, 2011, pp. 517-521, doi: 10.1109/ICCSCE.2011.6190580.

\[6\] A. Al-Omary, M. M. Akram and V. Dhamodharan, "Design and Implementation of Intelligent Socializing 3D Humanoid Robot," *2021 International Conference on Innovation and Intelligence for Informatics, Computing, and Technologies (3ICT)*, Zallaq, Bahrain, 2021, pp. 398-402, doi: 10.1109/3ICT53449.2021.9582077

\[7\] A. Poulose, J. H. Kim and D. S. Han, "Feature Vector Extraction Technique for Facial Emotion Recognition Using Facial Landmarks," *2021 International Conference on Information and Communication Technology Convergence (ICTC)*, Jeju Island, Korea, Republic of, 2021, pp. 1072-1076, doi: 10.1109/ICTC52510.2021.9620798.

\[8\] A. S. M. Ahsanul Sarkar Akib, M. Farhan Ferdous, M. Biswas and H. M. Khondokar, "Artificial Intelligence Humanoid BONGO Robot in Bangladesh," *2019 1st International Conference on Advances in Science, Engineering and Robotics Technology (ICASERT)*, Dhaka, Bangladesh, 2019, pp. 1-6, doi: 10.1109/ICASERT.2019.8934748.

\[9\] Kaneko, K., Kanehiro, F., Kajita, S., Yokoyama, K., Akachi, K., Kawasaki, T., Ota, S. and Isozumi, T., 2002, September. Design of prototype humanoid robotics platform for HRP. In *IEEE/RSJ International Conference on Intelligent Robots and Systems* (Vol. 3, pp. 2431-2436). IEEE.

\[10\] Yu, Z., Huang, Q., Ma, G., Chen, X., Zhang, W., Li, J. and Gao, J., 2014. Design and development of the humanoid robot BHR-5. *Advances in Mechanical Engineering*, *6*, p.852937.

\[11\] Nakadai, K., Matsui, T., Okuno, H.G. and Kitano, H., 2000, October. Active audition system and humanoid exterior design. In *Proceedings. 2000 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2000)(Cat. No. 00CH37113)* (Vol. 2, pp. 1453-1461). IEEE.

\[12\] Buschmann, T., Lohmeier, S. and Ulbrich, H., 2009. Humanoid robot lola: Design and walking control. *Journal of physiology-Paris*, *103*(3-5), pp.141-148.

\[13\] Chignoli, M., Kim, D., Stanger-Jones, E. and Kim, S., 2021, July. The MIT humanoid robot: Design, motion planning, and control for acrobatic behaviors. In *2020 IEEE-RAS 20th International Conference on Humanoid Robots (Humanoids)* (pp. 1-8). IEEE.

\[14\] Albers, A., Brudniok, S., Ottnad, J., Sauter, C. and Sedchaicharn, K., 2006, December. Upper body of a new humanoid robot-the design of ARMAR III. In *2006 6th IEEE-RAS International Conference on Humanoid Robots* (pp. 308-313). IEEE

\[15\] J. Hirth, N. Schmitz and K. Berns, "Emotional Architecture for the Humanoid Robot Head ROMAN," Proceedings 2007 IEEE International Conference on Robotics and Automation, Rome, Italy, 2007, pp. 2150-2155, doi: 10.1109/ROBOT.2007.363639.

\[16\] S. Fuenzalida, K. Toapanta, J. Paillacho and D. Paillacho, "Forward and Inverse Kinematics of a Humanoid Robot Head for Social Human Robot-Interaction," 2019 IEEE Fourth Ecuador Technical Chapters Meeting (ETCM), Guayaquil, Ecuador, 2019, pp. 1-4, doi: 10.1109/ETCM48019.2019.9014
