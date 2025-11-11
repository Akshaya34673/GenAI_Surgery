# 🩺 Surgical Instrument Detection and Scene Understanding using MViT
---

## 📘 Overview

This project is an **AI-powered web application** designed for **surgical instrument detection** and **surgical scene understanding** using the **Multiscale Vision Transformer (MViT)** model.  
It identifies instruments such as *Bipolar, Monopolar, Needle Driver, and Suction* from surgical videos or images.

The system integrates:
- 🧠 Deep Learning (MViT)
- ⚙️ Flask Backend (API + Model Serving)
- 💻 React Frontend (Interactive UI)
- 🗃️ MongoDB Atlas (User Data + History Storage)

---

## 🎯 Key Features

- 🔍 Detects multiple surgical instruments using MViT
- 🧩 Segmentation and phase prediction modules (extendable)
- 🔑 JWT-based User Authentication (Login & Signup)
- 💾 User-specific upload history from MongoDB
- 🖼️ Upload & visualize results directly on web
- 🧾 Clean, modular Flask REST API
- 🎨 Responsive UI with Tailwind CSS

---

## 🧠 Model: Multiscale Vision Transformer (MViT)

MViT divides an image into small patches and uses **multi-head self-attention** to learn features at multiple scales.

### 🧩 Architecture Summary
| Component | Function |
|------------|-----------|
| **PatchEmbed** | Splits the input image into small patches |
| **Pooling Attention** | Focuses on important image regions |
| **Patch Merging** | Reduces spatial size for multi-scale learning |
| **MViT Blocks** | Deep transformer layers with self-attention |
| **Classification Head** | Predicts probability of each surgical instrument |

### 🧠 Output Example
```json
{
  "Bipolar": 0.91,
  "NeedleDriver": 0.23,
  "Monopolar": 0.94,
  "Suction": 0.61
}
```
| Layer        | Technology                              |
| ------------ | --------------------------------------- |
| **Frontend** | React.js, Tailwind CSS                  |
| **Backend**  | Flask (Python)                          |
| **Model**    | Multiscale Vision Transformer (PyTorch) |
| **Database** | MongoDB Atlas                           |
| **Auth**     | JWT (JSON Web Tokens)                   |

<img width="515" height="617" alt="image" src="https://github.com/user-attachments/assets/b08ee06f-853c-46e7-bb85-ef4dc729021e" />
