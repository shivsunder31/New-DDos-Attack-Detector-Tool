
# DDoS Attack Detection Model
#  (Cyber security and Machine learning )
Link -  https://ddos-attack-detection.onrender.com

## Problem Statement

DDoS (Distributed Denial of Service) attacks are a serious threat to network security, where multiple compromised systems are used to target a single system, causing it to become unavailable to its intended users. Detecting and mitigating such attacks is crucial to ensuring the stability and availability of network services.

## What is DDoS?

DDoS (Distributed Denial of Service) is a malicious attempt to disrupt normal traffic of a targeted server, service, or network by overwhelming it with a flood of internet traffic. This is often achieved by using a large number of compromised devices (botnets) to send traffic to the target, thus causing it to become unreachable or sluggish for legitimate users.

## Solution

This project aims to develop a machine learning model for detecting DDoS attacks in network traffic. By analyzing various features of network traffic packets, such as packet counts, flow duration, flag counts, and activity metrics, we can train models to distinguish between legitimate and malicious traffic.

## How it Occurs

DDoS attacks typically occur when attackers exploit vulnerabilities in network protocols or applications to gain control over a large number of devices, forming a botnet. These compromised devices are then instructed to flood the target server or network with traffic, overwhelming its resources and causing it to become inaccessible to legitimate users.

## Model Overview

We utilized various machine learning algorithms including K-Nearest Neighbors (KNN), Random Forest, Gaussian Mixture Models (GMM), K-Means Clustering, and Support Vector Machines (SVM) to develop our DDoS detection model. After extensive experimentation and evaluation, we achieved an impressive accuracy rate of 99% on our model.

## Tech Stack

- **Python**: Primary programming language used for model development.
- **Flask**: Web framework for building the web server.
- **Pandas** and **NumPy**: Libraries for data manipulation and numerical computing.
- **Scikit-learn**: Machine learning library for model development and evaluation.
- **HTML**, **CSS**: Frontend technologies for building the user interface.
- **GitHub** and **Render**: Platforms used for version control and deployment, respectively.
  

## Project Workflow

1. **Data Collection**: Gathering network traffic data for model training.
2. **Data Preprocessing**: Cleaning, filtering, and transforming the data to prepare it for modeling.
3. **Feature Engineering**: Selecting and engineering relevant features from the dataset.
4. **Model Selection**: Choosing appropriate machine learning algorithms for the task.
5. **Model Training and Evaluation**: Training the selected models on the data and evaluating their performance.
6. **Web Server Development**: Building the Flask-based web server to host the model.
7. **User Interface Design**: Designing the frontend interface for interacting with the model.
8. **Integration**: Integrating the model with the web server and user interface.
9. **Deployment**: Deploying the application to a web server for public access.



TCP, ICMP, and UDP protocols all play important roles in detecting Distributed Denial of Service (DDoS) attacks, each in its own way:

1. **TCP**:  (Transmission control protocol)
   - DDoS attacks often involve overwhelming a server or network with a high volume of TCP connection requests.
   - Monitoring TCP connections and their states can help detect abnormal behavior such as a sudden spike in connection attempts or a large number of connections in the SYN_SENT state (indicating a TCP SYN flood attack).
   - Analyzing TCP header information, such as source IP addresses, destination ports, and flags, can provide insights into potential attack patterns.

2. **ICMP**: (Internet control message protocol)
   - ICMP packets can be used in DDoS attacks, such as ICMP flood attacks or ICMP echo request (ping) floods.
   - Monitoring ICMP traffic volume and analyzing ICMP packet characteristics can help detect abnormal patterns indicative of an ongoing attack.
   - Additionally, monitoring ICMP error messages, such as ICMP Destination Unreachable or ICMP Time Exceeded, can provide insights into potential network issues caused by DDoS attacks.

3. **UDP**:  (User datagram protocol)
   - UDP-based DDoS attacks involve flooding a target server or network with a high volume of UDP packets, often targeting specific UDP ports used by vulnerable services.
   - Monitoring UDP traffic and analyzing packet characteristics, such as source IP addresses, destination ports, and payload patterns, can help detect UDP flood attacks or other UDP-based DDoS techniques.
   - Additionally, monitoring UDP-based services for abnormal behavior, such as excessive packet loss or increased response times, can aid in detecting UDP-based DDoS attacks targeting specific applications or services.

In summary, monitoring network traffic and analyzing the behavior of TCP, ICMP, and UDP protocols can help detect DDoS attacks by identifying abnormal patterns, such as unusually high traffic volume, unexpected packet characteristics, or unusual network behavior. By leveraging insights from these protocols, network administrators can implement effective DDoS detection and mitigation strategies to protect their networks and services.
