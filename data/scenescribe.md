# SceneScribe — AI smart glasses for the visually impaired (Mechatronics NUST project)

SceneScribe is a pair of AI-enabled intelligent glasses for visually impaired people, developed as a final-year design project at the Department of Mechatronics Engineering, NUST College of EME, and on display at the department's stall at Indus RAS Expo 2026. The lightweight smart glasses describe the wearer's surroundings, answer spoken questions, guide them to destinations, and warn of obstacles through vibration — all running fully offline on an NVIDIA Jetson Orin Nano, with no internet needed. The 2026 version was built by Fatima Sarwar and Baseer Ayaz Raja, supervised by Dr Ayesha Zeb with co-supervisors Dr Umar Shahbaz Khan and Senior Lecturer Usman Asad. It builds on the original cloud-based SceneScribe prototype by Zaid Mahboob and Salman Sadiq.

> Source: SceneScribe final-year project thesis, NUST CEME, 2026

## Why SceneScribe was built — assisting blind and visually impaired people

According to the World Health Organization, about 285 million people worldwide are visually impaired, including roughly 1.5 million in Pakistan. Vision loss restricts independent movement and creates dependence on others in crowded or unfamiliar places. SceneScribe restores autonomy by giving the wearer a spoken understanding of their surroundings and safe, step-by-step navigation — combining scene description, navigation and obstacle avoidance in one wearable device instead of a bulky headset, separate smart cane and speakers.

## How SceneScribe works — offline AI on smart glasses

The glasses integrate a camera, microphone, speakers and vibration motors, powered by a Jetson Orin Nano edge computer. Everything runs offline: the wearer says the wake word "Hey Glasses" (detected by Vosk), speech is transcribed locally with Faster Whisper, a compact vision-language model (qwen3-vl 2B instruct, selected after benchmarking against alternatives like Gemma 3 for latency, throughput, power and hallucination resistance) interprets the scene and the question, and Piper text-to-speech speaks the answer through the built-in speakers. Navigation uses Dijkstra's algorithm for global pathfinding with ArUco markers and a YOLO scene classifier for localization; obstacle avoidance uses monocular depth estimation with instance segmentation, triggering haptic vibration alerts when obstacles are close.

## SceneScribe performance and user testing

The YOLO scene classifier identified locations correctly 98% of the time across 10,422 test images at 75 frames per second; the combined localization system was 91% accurate across 9,340 images at 29 frames per second in real time. Moving from cloud to offline processing cut speech-to-text delay from 13.4 seconds to 0.67 seconds and end-to-end response time from 30–35 seconds to 20–25 seconds, while removing internet dependence entirely. The prototype was validated in direct testing with visually impaired users, whose feedback (on ergonomics, weight distribution and audio timing) is shaping the next design. The related utility patent, "An Adaptive AI-enabled System for Navigation of Visually Impaired Persons", was filed in June 2024, and the project received PKR 1 million in NUST funding.
