# HARP — Humanoid Assistive Robotic Platform

HARP (Humanoid Assistive Robotic Platform) is a humanoid assistive robot developed at the Department of Mechatronics Engineering, NUST College of Electrical and Mechanical Engineering (CEME), Rawalpindi, by the Industrial AI and IoT (IAI) Research Group. It is a low-cost, modular humanoid that can see, listen, talk and move in indoor service environments — designed for reception, guidance and assistance duties, speaking both English and Urdu. The project was funded by NUST (Flagship scheme, PKR 1 million), approved 17 November 2023 and completed 30 December 2025.

> Source: HARP project completion report (December 2025)

## Who built HARP — project team

- Principal Investigator: Dr Kanwal Naveed, CEME, NUST
- Co-Principal Investigators: Dr Tahir Habib Nawaz and Senior Lecturer Usman Asad, CEME, NUST
- Other team members: Dr Muhammad Moazam Fraz (SEECS, NUST — AI and machine learning modules) and Dr Anas Bin Aqeel (CEME — mechanical design and system integration co-supervision)
- Research assistants Kashaan Ansari and Ahyan Ahmed, plus three final-year design project teams of the Department of Mechatronics Engineering (2023-24, 2024-25, 2025-26) and five undergraduate interns

The robot lives in the Robot Design and Development Lab (RDDL) of the National Centre of Robotics and Automation (NCRA) at NUST CEME.

## How HARP works — technology

HARP follows an edge-computing design: perception runs on small onboard computers (Raspberry Pi 5, and earlier a Raspberry Pi 4B and NVIDIA Jetson Nano), with cloud AI used only where it adds clear value, such as large-language-model conversation. ROS 2 connects the sensors and modules. The robot navigates with cameras instead of LiDAR, using an Intel RealSense D435i RGB-D depth camera with RTAB-Map for simultaneous localization and mapping (SLAM) and A* path planning, on a custom omnidirectional base with four mecanum wheels that can move in any direction, including sideways. Its head has a 2-degree-of-freedom neck, active head tracking, and an animated expressive face with eyes that blink and change expression. Perception includes face detection with depth, gender, emotion and gesture recognition, human behavior recognition, and object detection and tracking.

## How HARP talks — conversation and languages

HARP's conversational AI is powered by large language models with real-time voice: it listens and replies in natural spoken English and Urdu. Its answers about the venue and event are grounded in a local knowledge base through retrieval (RAG), so it answers from its institution's actual documents rather than guessing. The conversation system evolved from a scripted chatbot, to a retrieval-grounded pipeline, to full-duplex native-audio realtime models, with live subtitles shown on its dashboard for accessibility. HARP greets known people by name, and for privacy it never stores the faces of unknown people.

## Mechatronics stall at Indus RAS Expo 2026 — projects on display

At the Department of Mechatronics Engineering stall at Indus RAS Expo 2026, HARP itself is on display along with two other final-year projects: ADR, the Autonomous Disinfectant Robot for hospitals (a mecanum-wheel robot that mops, vacuums and fumigates autonomously), and SceneScribe, AI-enabled smart glasses that describe surroundings, navigate and warn of obstacles for visually impaired people, running fully offline.

## Why HARP was built — purpose and objectives

Project objectives: develop a humanoid assistive robot with robust perception, seamless human-robot interaction, and safe autonomous mobility for real-world service environments; deliver consistent, personalized, uninterrupted service through AI-driven learning; and integrate natural language processing, computer vision and machine learning into a scalable, cost-effective solution. The motivation includes rising demand for service and care assistance — the WHO projects a global shortfall of roughly ten million health workers by 2030 — and the value of contactless service in settings like hospitals, reception desks and inquiry counters, particularly in Pakistan. The project also served as a capacity-building vehicle, with successive student teams extending the platform at NUST.
