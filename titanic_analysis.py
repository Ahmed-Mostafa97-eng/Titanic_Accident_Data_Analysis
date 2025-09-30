#!/usr/bin/env python3
"""
Titanic Survival Analysis
========================

A comprehensive exploratory data analysis of the Titanic passenger dataset
to understand factors affecting survival rates during the disaster.

Author: Ahmed Mostafa
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

class TitanicAnalyzer:
    """
    A comprehensive analyzer for Titanic passenger survival data.
    
    This class provides methods for data cleaning, exploratory data analysis,
    and visualization of survival patterns.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data path.
        
        Args:
            data_path (str): Path to the Titanic dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load the Titanic dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def explore_data_structure(self):
        """Perform initial data exploration and display basic information."""
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return
        
        print("\n" + "="*60)
        print("📊 DATASET OVERVIEW")
        print("="*60)
        
        # Basic information
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Data types and unique values
        print("\n📋 Column Information:")
        info_df = pd.DataFrame({
            "Data Type": self.df.dtypes,
            "Non-Null Count": self.df.count(),
            "Null Count": self.df.isnull().sum(),
            "Null Percentage": (self.df.isnull().sum() / len(self.df) * 100).round(2),
            "Unique Values": self.df.nunique()
        })
        print(info_df)
        
        # Display first few rows
        print("\n📝 Sample Data:")
        print(self.df.head())
        
        return info_df
    
    def clean_data(self):
        """
        Perform comprehensive data cleaning based on analysis findings.
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        print("\n" + "="*60)
        print("🧹 DATA CLEANING PROCESS")
        print("="*60)
        
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # 1. Remove duplicates
        initial_rows = len(self.df_clean)
        self.df_clean.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(self.df_clean)
        print(f"🔄 Duplicates removed: {duplicates_removed}")
        
        # 2. Convert data types for categorical variables
        categorical_cols = ["Pclass", "Sex", "Embarked"]
        for col in categorical_cols:
            if col in self.df_clean.columns:
                self.df_clean[col] = self.df_clean[col].astype("category")
        print(f"📊 Converted to categorical: {categorical_cols}")
        
        # 3. Handle missing values
        print("\n🔍 Handling Missing Values:")
        
        # Age: Fill with median grouped by Pclass and Sex
        if "Age" in self.df_clean.columns:
            age_median = self.df_clean.groupby(["Pclass", "Sex"])["Age"].transform("median")
            age_nulls = self.df_clean["Age"].isnull().sum()
            self.df_clean["Age"].fillna(age_median, inplace=True)
            print(f"   • Age: {age_nulls} missing values filled with group median")
        
        # Embarked: Fill with mode
        if "Embarked" in self.df_clean.columns:
            embarked_nulls = self.df_clean["Embarked"].isnull().sum()
            if embarked_nulls > 0:
                mode_embarked = self.df_clean["Embarked"].mode()[0]
                self.df_clean["Embarked"].fillna(mode_embarked, inplace=True)
                print(f"   • Embarked: {embarked_nulls} missing values filled with mode ({mode_embarked})")
        
        # Cabin: Drop due to high missing percentage (>70%)
        if "Cabin" in self.df_clean.columns:
            cabin_nulls = self.df_clean["Cabin"].isnull().sum()
            cabin_null_pct = (cabin_nulls / len(self.df_clean)) * 100
            if cabin_null_pct > 70:
                self.df_clean.drop(columns=["Cabin"], inplace=True)
                print(f"   • Cabin: Column dropped ({cabin_null_pct:.1f}% missing)")
        
        # 4. Feature Engineering
        print("\n🔧 Feature Engineering:")
        
        # Family size
        if "SibSp" in self.df_clean.columns and "Parch" in self.df_clean.columns:
            self.df_clean["FamilySize"] = self.df_clean["SibSp"] + self.df_clean["Parch"] + 1
            self.df_clean["IsAlone"] = (self.df_clean["FamilySize"] == 1).astype(int)
            print("   • Created FamilySize and IsAlone features")
        
        # Title extraction from Name
        if "Name" in self.df_clean.columns:
            self.df_clean["Title"] = self.df_clean["Name"].str.extract(" ([A-Za-z]+)\\. ", expand=False)
            # Group rare titles
            title_counts = self.df_clean["Title"].value_counts()
            rare_titles = title_counts[title_counts < 10].index
            self.df_clean["Title"] = self.df_clean["Title"].replace(rare_titles, "Rare")
            self.df_clean["Title"] = self.df_clean["Title"].astype("category")
            print("   • Extracted and categorized titles from names")
        
        # Age groups
        if "Age" in self.df_clean.columns:
            self.df_clean["AgeGroup"] = pd.cut(self.df_clean["Age"], 
                                             bins=[0, 12, 18, 35, 60, 100], 
                                             labels=["Child", "Teen", "Adult", "Middle-aged", "Senior"])
            print("   • Created age groups")
        
        # Fare groups
        if "Fare" in self.df_clean.columns:
            self.df_clean["FareGroup"] = pd.qcut(self.df_clean["Fare"], 
                                               q=4, 
                                               labels=["Low", "Medium", "High", "Very High"])
            print("   • Created fare quartile groups")
        
        print(f"\n✅ Data cleaning completed. Final shape: {self.df_clean.shape}")
        
        # Final check for missing values
        remaining_nulls = self.df_clean.isnull().sum().sum()
        if remaining_nulls == 0:
            print("✅ No missing values remaining")
        else:
            print(f"⚠️  {remaining_nulls} missing values still present")
        
        return self.df_clean
    
    def analyze_survival_patterns(self):
        """Analyze survival patterns across different passenger characteristics."""
        if self.df_clean is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return
        
        print("\n" + "="*60)
        print("📈 SURVIVAL PATTERN ANALYSIS")
        print("="*60)
        
        # Overall survival rate
        survival_rate = self.df_clean["Survived"].mean() * 100
        print(f"🎯 Overall Survival Rate: {survival_rate:.1f}%")
        
        # Survival by key features
        categorical_features = ["Sex", "Pclass", "Embarked", "AgeGroup", "Title"]
        
        survival_analysis = {}
        
        for feature in categorical_features:
            if feature in self.df_clean.columns:
                survival_by_feature = self.df_clean.groupby(feature)["Survived"].agg(["count", "sum", "mean"])
                survival_by_feature.columns = ["Total", "Survived_Count", "Survival_Rate"]
                survival_by_feature["Survival_Rate"] = survival_by_feature["Survival_Rate"] * 100
                survival_analysis[feature] = survival_by_feature
                
                print(f"\n📊 Survival by {feature}:")
                print(survival_by_feature.round(1))
        
        return survival_analysis
    
    def create_visualizations(self, save_path: str = None):
        """
        Create comprehensive visualizations for the analysis.
        
        Args:
            save_path (str): Directory to save the plots
        """
        if self.df_clean is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return
        
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting environment
        plt.rcParams["figure.figsize"] = (15, 10)
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall survival distribution
        plt.subplot(4, 3, 1)
        survival_counts = self.df_clean["Survived"].value_counts()
        colors = ["#ff6b6b", "#4ecdc4"]
        plt.pie(survival_counts.values, labels=["Did not survive", "Survived"], 
                autopct="%1.1f%%", colors=colors, startangle=90)
        plt.title("Overall Survival Distribution", fontsize=14, fontweight="bold")
        
        # 2. Survival by Gender
        plt.subplot(4, 3, 2)
        survival_by_sex = pd.crosstab(self.df_clean["Sex"], self.df_clean["Survived"], normalize="index") * 100
        survival_by_sex.plot(kind="bar", color=colors, ax=plt.gca())
        plt.title("Survival Rate by Gender", fontsize=14, fontweight="bold")
        plt.ylabel("Survival Rate (%)")
        plt.legend(["Did not survive", "Survived"])
        plt.xticks(rotation=0)
        
        # 3. Survival by Passenger Class
        plt.subplot(4, 3, 3)
        survival_by_class = pd.crosstab(self.df_clean["Pclass"], self.df_clean["Survived"], normalize="index") * 100
        survival_by_class.plot(kind="bar", color=colors, ax=plt.gca())
        plt.title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
        plt.ylabel("Survival Rate (%)")
        plt.legend(["Did not survive", "Survived"])
        plt.xticks(rotation=0)
        
        # 4. Age distribution by survival
        plt.subplot(4, 3, 4)
        survived = self.df_clean[self.df_clean["Survived"] == 1]["Age"]
        not_survived = self.df_clean[self.df_clean["Survived"] == 0]["Age"]
        plt.hist([not_survived, survived], bins=30, alpha=0.7, 
                label=["Did not survive", "Survived"], color=colors)
        plt.title("Age Distribution by Survival", fontsize=14, fontweight="bold")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.legend()
        
        # 5. Fare distribution by survival
        plt.subplot(4, 3, 5)
        survived_fare = self.df_clean[self.df_clean["Survived"] == 1]["Fare"]
        not_survived_fare = self.df_clean[self.df_clean["Survived"] == 0]["Fare"]
        plt.hist([not_survived_fare, survived_fare], bins=30, alpha=0.7,
                label=["Did not survive", "Survived"], color=colors)
        plt.title("Fare Distribution by Survival", fontsize=14, fontweight="bold")
        plt.xlabel("Fare")
        plt.ylabel("Frequency")
        plt.legend()
        plt.xlim(0, 200)  # Limit x-axis for better visibility
        
        # 6. Survival by Embarked Port
        plt.subplot(4, 3, 6)
        survival_by_embarked = pd.crosstab(self.df_clean["Embarked"], self.df_clean["Survived"], normalize="index") * 100
        survival_by_embarked.plot(kind="bar", color=colors, ax=plt.gca())
        plt.title("Survival Rate by Embarked Port", fontsize=14, fontweight="bold")
        plt.ylabel("Survival Rate (%)")
        plt.legend(["Did not survive", "Survived"])
        plt.xticks(rotation=0)
        
        # 7. Family Size vs Survival
        plt.subplot(4, 3, 7)
        if "FamilySize" in self.df_clean.columns:
            survival_by_family = pd.crosstab(self.df_clean["FamilySize"], self.df_clean["Survived"], normalize="index") * 100
            survival_by_family.plot(kind="bar", color=colors, ax=plt.gca())
            plt.title("Survival Rate by Family Size", fontsize=14, fontweight="bold")
            plt.ylabel("Survival Rate (%)")
            plt.legend(["Did not survive", "Survived"])
            plt.xticks(rotation=0)
        
        # 8. Age Group vs Survival
        plt.subplot(4, 3, 8)
        if "AgeGroup" in self.df_clean.columns:
            survival_by_age_group = pd.crosstab(self.df_clean["AgeGroup"], self.df_clean["Survived"], normalize="index") * 100
            survival_by_age_group.plot(kind="bar", color=colors, ax=plt.gca())
            plt.title("Survival Rate by Age Group", fontsize=14, fontweight="bold")
            plt.ylabel("Survival Rate (%)")
            plt.legend(["Did not survive", "Survived"])
            plt.xticks(rotation=45)
        
        # 9. Correlation Heatmap
        plt.subplot(4, 3, 9)
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df_clean[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, 
                   square=True, ax=plt.gca())
        plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        
        # 10. Survival by Title
        plt.subplot(4, 3, 10)
        if "Title" in self.df_clean.columns:
            survival_by_title = pd.crosstab(self.df_clean["Title"], self.df_clean["Survived"], normalize="index") * 100
            survival_by_title.plot(kind="bar", color=colors, ax=plt.gca())
            plt.title("Survival Rate by Title", fontsize=14, fontweight="bold")
            plt.ylabel("Survival Rate (%)")
            plt.legend(["Did not survive", "Survived"])
            plt.xticks(rotation=45)
        
        # 11. Passenger Class and Gender Interaction
        plt.subplot(4, 3, 11)
        class_sex_survival = self.df_clean.groupby(["Pclass", "Sex"])["Survived"].mean() * 100
        class_sex_survival.unstack().plot(kind="bar", color=["#ff9999", "#66b3ff"], ax=plt.gca())
        plt.title("Survival Rate by Class and Gender", fontsize=14, fontweight="bold")
        plt.ylabel("Survival Rate (%)")
        plt.legend(["Female", "Male"])
        plt.xticks(rotation=0)
        
        # 12. Summary Statistics Box
        plt.subplot(4, 3, 12)
        plt.axis("off")
        
        # Calculate key statistics
        total_passengers = len(self.df_clean)
        survivors = self.df_clean["Survived"].sum()
        survival_rate = (survivors / total_passengers) * 100
        avg_age = self.df_clean["Age"].mean()
        avg_fare = self.df_clean["Fare"].mean()
        female_survival_rate = self.df_clean[self.df_clean['Sex'] == 'female']['Survived'].mean()*100
        class1_survival_rate = self.df_clean[self.df_clean['Pclass'] == 1]['Survived'].mean()*100
        male_survival_rate = self.df_clean[self.df_clean['Sex'] == 'male']['Survived'].mean()*100
        class3_survival_rate = self.df_clean[self.df_clean['Pclass'] == 3]['Survived'].mean()*100

        stats_text = f"""
        📊 KEY STATISTICS
        
        Total Passengers: {total_passengers:,}
        Survivors: {survivors:,}
        Overall Survival Rate: {survival_rate:.1f}%
        
        Average Age: {avg_age:.1f} years
        Average Fare: ${avg_fare:.2f}
        
        Highest Survival Rate:
        • Women: {female_survival_rate:.1f}%
        • 1st Class: {class1_survival_rate:.1f}%
        
        Lowest Survival Rate:
        • Men: {male_survival_rate:.1f}%
        • 3rd Class: {class3_survival_rate:.1f}%
        """
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot if path is provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_path}/titanic_comprehensive_analysis.png", 
                       dpi=300, bbox_inches="tight")
            print(f"📊 Comprehensive analysis plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_insights(self):
        """Generate key insights from the analysis."""
        if self.df_clean is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return
        
        print("\n" + "="*60)
        print("💡 KEY INSIGHTS AND FINDINGS")
        print("="*60)
        
        insights = []
        
        # Gender analysis
        female_survival = self.df_clean[self.df_clean["Sex"] == "female"]["Survived"].mean() * 100
        male_survival = self.df_clean[self.df_clean["Sex"] == "male"]["Survived"].mean() * 100
        insights.append(f"🚺 Women had a {female_survival:.1f}% survival rate vs {male_survival:.1f}% for men")
        
        # Class analysis
        class_survival = self.df_clean.groupby("Pclass")["Survived"].mean() * 100
        insights.append(f"🎭 1st class passengers had {class_survival[1]:.1f}% survival rate vs {class_survival[3]:.1f}% for 3rd class")
        
        # Age analysis
        if "AgeGroup" in self.df_clean.columns:
            age_survival = self.df_clean.groupby("AgeGroup")["Survived"].mean() * 100
            child_survival = age_survival.get("Child", 0)
            if child_survival > 0:
                insights.append(f"👶 Children had a {child_survival:.1f}% survival rate")
        
        # Family size analysis
        if "IsAlone" in self.df_clean.columns:
            alone_survival = self.df_clean[self.df_clean["IsAlone"] == 1]["Survived"].mean() * 100
            family_survival = self.df_clean[self.df_clean["IsAlone"] == 0]["Survived"].mean() * 100
            insights.append(f"👨‍👩‍👧‍👦 Passengers with family had {family_survival:.1f}% survival rate vs {alone_survival:.1f}% for solo travelers")
        
        # Fare analysis
        high_fare_survival = self.df_clean[self.df_clean["Fare"] > self.df_clean["Fare"].median()]["Survived"].mean() * 100
        low_fare_survival = self.df_clean[self.df_clean["Fare"] <= self.df_clean["Fare"].median()]["Survived"].mean() * 100
        insights.append(f"💰 Higher fare passengers had {high_fare_survival:.1f}% survival rate vs {low_fare_survival:.1f}% for lower fare")
        
        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # Additional statistical insights
        print(f"\n📈 Statistical Observations:")
        print(f"   • Total passengers analyzed: {len(self.df_clean):,}")
        print(f"   • Overall survival rate: {self.df_clean['Survived'].mean()*100:.1f}%")
        print(f"   • Average age: {self.df_clean['Age'].mean():.1f} years")
        print(f"   • Age range: {self.df_clean['Age'].min():.0f} - {self.df_clean['Age'].max():.0f} years")
        print(f"   • Fare range: ${self.df_clean['Fare'].min():.2f} - ${self.df_clean['Fare'].max():.2f}")
        
        return insights
    
    def save_cleaned_data(self, output_path: str):
        """Save the cleaned dataset to a CSV file."""
        if self.df_clean is None:
            print("❌ No cleaned data available. Please run clean_data() first.")
            return
        
        try:
            self.df_clean.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")


def main():
    """Main function to run the complete analysis."""
    print("🚢 TITANIC SURVIVAL ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    data_path = "../data/titanic.csv"
    analyzer = TitanicAnalyzer(data_path)
    
    # Run complete analysis pipeline
    try:
        # Load and explore data
        analyzer.load_data()
        analyzer.explore_data_structure()
        
        # Clean data
        analyzer.clean_data()
        
        # Analyze patterns
        analyzer.analyze_survival_patterns()
        
        # Create visualizations
        analyzer.create_visualizations(save_path="../results/figures")
        
        # Generate insights
        analyzer.generate_insights()
        
        # Save cleaned data
        analyzer.save_cleaned_data("../data/titanic_cleaned.csv")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()

