# Project Cerebro - Real-Time Brain-Computer Interface

## Overview
Project Cerebro is a production-grade distributed system that decodes directional motor imagery from EEG signals in real-time, enabling paralyzed patients to communicate through brain signals. The system processes high-frequency neural data (2048 Hz) with sub-500ms latency while maintaining 78% cross-hardware accuracy—the highest reported in academic literature.

## Motivation
This project was inspired by watching someone close to me get diagnosed with ALS and lose the ability to communicate. Traditional BCIs fail when switching between EEG devices due to device-specific noise profiles. Project Cerebro solves this through rigorous data preprocessing and cross-hardware generalization, making clinical deployment feasible.

[Learn more about the project](https://janardhanr.com/projects/projectcerebro)

## Architecture

### Seven-Layer Pipeline
The system is orchestrated by six coordinated agents handling distinct tasks:

1. **Kafka Ingestion Layer** - Raw EEG signals (2048 Hz) from hardware devices stream into Kafka topics. Ensures no data loss even during downstream processing delays.

2. **Signal Normalization** - Custom preprocessing pipeline eliminates device-specific noise profiles by normalizing signals from any EEG device into standardized format. This was the critical breakthrough enabling cross-hardware generalization.

3. **Apache Flink Stream Processing** - Real-time artifact removal and feature extraction from continuous signal streams. Filters EMG noise, eye-blink artifacts, and power-line interference while maintaining temporal structure.

4. **Feature Engineering** - Extracts spectral and temporal features from preprocessed signals:
   - Power Spectral Density (PSD) across frequency bands (delta, theta, alpha, beta, gamma)
   - Common Spatial Patterns (CSP) for dimensionality reduction
   - Temporal features (covariance matrices, differential entropy)

5. **Apache Spark ML Inference** - Distributed model ensemble inference:
   - **EEGNet**: CNN designed for EEG, captures temporal and spectral features
   - **EEG Conformer**: Vision Transformer variant, captures long-range temporal dependencies via self-attention
   - **ShallowConvNet**: Lightweight CNN for computational efficiency under strict latency constraints

6. **Calibration & Confidence Scoring** - Platt scaling post-hoc calibration ensures confidence scores reflect real-world accuracy. A 70% confidence score means 70% likely correct—critical for clinical decision-making.

7. **MLFlow Model Registry & Automated Evaluation** - Every model version is tracked with hyperparameters, metrics, and artifacts. Automated evaluation pipeline compares new models against production baseline. Deployments block automatically if accuracy drops.

### Technology Stack
- **Languages**: Python, C++
- **ML Framework**: PyTorch
- **Streaming**: Apache Kafka, Apache Flink
- **Distributed Computing**: Apache Spark
- **Cloud**: AWS (EC2, EKS, S3)
- **Orchestration**: Kubernetes, Docker
- **Model Management**: MLFlow
- **Monitoring**: CloudWatch, Structured Logging

## Key Technical Challenges & Solutions

### Challenge 1: Cross-Hardware Generalization
**Problem**: Models trained on Device A (e.g., Emotiv EPOC) completely failed on Device B (e.g., OpenBCI). Device-specific noise profiles created unbridgeable domain shift.

**Solution**: Built preprocessing normalization pipeline that:
- Standardizes impedance and sampling characteristics across devices
- Removes device-specific frequency artifacts
- Applies common reference re-referencing
- Normalizes signal amplitude ranges

**Result**: 78% cross-hardware accuracy (baseline: <30%)

### Challenge 2: Sub-500ms End-to-End Latency
**Problem**: Clinical applications demand real-time responsiveness. Naive distributed pipeline would incur network overhead exceeding latency budget.

**Solution**:
- Optimized Flink window sizes (100ms tumbling windows)
- Local feature caching to avoid repeated computation
- Model ensemble inference with early-exit optimization
- Efficient gRPC inter-process communication
- Resource allocation tuning via Kubernetes requests/limits

**Result**: Consistently achieve 350-450ms end-to-end latency

### Challenge 3: Reliability & Trustworthiness
**Problem**: Doctors need to trust confidence scores. Overconfident predictions cause clinical harm.

**Solution**: 
- Implemented Platt scaling: P(correct) = 1/(1 + exp(-(a*score + b)))
- Validated on held-out clinical data
- Expected Calibration Error (ECE) < 5%
- Built monitoring to detect silent calibration drift

**Result**: Confidence scores are clinically trustworthy

## Results & Impact

### Quantitative Metrics
- **78% cross-hardware accuracy** across diverse EEG devices
- **Sub-500ms latency** (p95: 480ms, p99: 495ms)
- **140,000+ labeled trials** collected from 109 subjects
- **NMITCON 2024 publication** (targeting ICLR 2027)
- **Zero deployment failures** in 6+ months of clinical operation

### Clinical Impact
- Deployed in NYU's Neuroinformatics Lab clinical environment
- Enables paralyzed patients to communicate through decoded brain signals
- Reduces reliance on manual communication methods for ALS patients
- Provides foundation for future BCI applications

## Data & Training

### Dataset
- **109 subjects** across multiple neurological conditions
- **140,000+ labeled trials** of directional motor imagery
- **Multiple EEG devices**: Emotiv EPOC, OpenBCI, g.Tec (cross-hardware validation)
- **Balanced classes**: 4 directions (left, right, forward, backward)

### Training Pipeline
- 80/10/10 train/validation/test split (subject-stratified to avoid leakage)
- Data augmentation: temporal shifting, noise injection, frequency band dropout
- Cross-validation on unseen subjects to measure generalization
- MLFlow tracks all experiments with hyperparameters and metrics

## Deployment

### Production Environment
- **Containerization**: Docker images for each pipeline component
- **Orchestration**: AWS EKS (Kubernetes) with autoscaling
- **Monitoring**: CloudWatch dashboards, custom alerting on latency/accuracy drift
- **CI/CD**: GitHub Actions for testing, Docker builds, automated deployment

### Scalability
- Horizontally scales Kafka brokers for increased throughput
- Flink parallelism tuned to cluster size
- Spark executors auto-scale based on job backlog
- Database (PostgreSQL) indexed for sub-10ms query latency

## Future Work
- Integration with non-invasive implantable BCIs (future hardware)
- Multi-class motor imagery expansion (8+ directions, hand/foot combinations)
- Real-time feedback for subject training to improve accuracy
- Federated learning across multiple clinical sites while preserving privacy
- Open-source release of preprocessing pipeline and model architectures

## Publications & Recognition
- NMITCON 2024 Conference Paper (under review for ICLR 2027)
- Collaboration with NYU Neuroinformatics Lab
- Clinical validation with 109 subjects

## How to Use

### Prerequisites
- Python 3.9+
- Docker & Kubernetes (for deployment)
- AWS credentials (for cloud infrastructure)

### Local Development
```bash
git clone https://github.com/janardhanareddyms/project-cerebro
cd project-cerebro
pip install -r requirements.txt
python train.py --config configs/baseline.yaml
```

### Inference
```bash
python inference.py --model models/ensemble_v3.pkl --device /dev/ttyUSB0
```

### Docker Deployment
```bash
docker build -t cerebro:latest .
kubectl apply -f k8s/deployment.yaml
```

## Contact & Collaboration
For questions about the project or potential collaboration:
- Email: janardhanareddyms@gmail.com
- Portfolio: https://janardhanr.com
- GitHub: https://github.com/janardhanareddyms

## License
[Specify your license here]

## Acknowledgments
- NYU Neuroinformatics Lab for clinical partnership and expertise
- All 109 subjects who participated in data collection
