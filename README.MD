# AI-Powered Customer Review Intelligence and Product Recommendation System

## Overview
This project analyzes e-commerce customer reviews using sentiment analysis and recommendation scoring to support marketing decisions and product targeting.

It was developed as a practical portfolio project based on research into recommendation systems, sentiment analysis, and their impact on customer satisfaction and marketing.

## Business Problem
E-commerce platforms receive large volumes of customer reviews, but raw review text is difficult to use directly for decision-making.

This project transforms customer feedback into structured insights by:
- cleaning raw review data
- analyzing customer sentiment
- aggregating product-level metrics
- generating recommendation scores
- visualizing the results

## Project Workflow
1. Raw review dataset loaded from CSV
2. Data cleaned and reduced to essential columns
3. Sentiment analysis applied to customer review text
4. Product-level metrics generated
5. Recommendation scores calculated
6. Charts created for business interpretation

## Tools Used
- Python
- pandas
- vaderSentiment
- matplotlib
- Oracle SQL
- Azure
- VS Code

## Project Structure
```text
amazon-ai-project
├── data
│   ├── raw
│   └── processed
├── scripts
│   ├── clean_data.py
│   ├── sentiment.py
│   ├── recommendation.py
│   └── visualize.py
├── sql
│   ├── schema.sql
│   └── queries.sql
├── outputs
├── shell
│   └── run_pipeline.bat
├── images
├── README.md
└── project_plan.md


Key Outputs
1. Sentiment Analysis Output

Customer reviews were analyzed and labeled as positive, negative, or neutral.

2. Recommendation Results

Products were ranked using a combined recommendation score based on:

average rating
average sentiment
review count

3. Sentiment Distribution Chart

This chart shows the overall distribution of customer sentiment.

4. Top Recommended Products

This chart shows the top 10 recommended products based on the scoring model.

Recommendation Logic

The recommendation score was calculated using:

40% average rating
40% average sentiment
20% review count score

This creates a simple and explainable ranking model for identifying products suitable for promotion and improved targeting.

Business Value

This project demonstrates how unstructured customer feedback can be converted into actionable business intelligence.

Possible use cases:

identifying high-potential products for promotion
detecting products with negative customer sentiment
supporting customer satisfaction analysis
improving data-driven marketing decisions
Future Improvements
connect outputs to Oracle SQL for deeper querying
integrate Azure cloud storage/database
enhance recommendation logic with collaborative filtering
build a dashboard for interactive analysis