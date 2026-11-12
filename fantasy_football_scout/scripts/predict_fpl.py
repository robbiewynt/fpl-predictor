"""
FPL Prediction Inference Script

This script shows how to use trained models to make predictions on new data.
Run after training models with fpl_model_training_complete.ipynb
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_model_artifacts(position, horizon='gw1'):
    """
    Load trained model, scaler, and feature list for a position and horizon.
    
    Args:
        position: 'Goalkeeper', 'Defender', 'Midfielder', or 'Forward'
        horizon: 'gw1', 'gw2', or 'gw3'
    
    Returns:
        model, scaler (or None), feature_list
    """
    models_dir = Path('../models')
    position_lower = position.lower()
    
    # Find model file (could be any algorithm)
    model_files = list(models_dir.glob(f"{position_lower}_{horizon}_*.joblib"))
    model_files = [f for f in model_files if 'scaler' not in f.name]
    
    if not model_files:
        raise FileNotFoundError(f"No model found for {position} {horizon}")
    
    model_file = model_files[0]
    model_name = model_file.stem.split('_')[-1]
    
    # Load model
    model = joblib.load(model_file)
    print(f"✓ Loaded {position} {horizon} model: {model_name}")
    
    # Load scaler if it exists (for linear models)
    scaler_file = models_dir / f"{position_lower}_{horizon}_scaler.joblib"
    scaler = joblib.load(scaler_file) if scaler_file.exists() else None
    if scaler:
        print(f"  ✓ Loaded scaler")
    
    # Load feature list
    features_file = models_dir / f"{position_lower}_{horizon}_features.txt"
    with open(features_file, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    print(f"  ✓ Loaded {len(features)} features")
    
    return model, scaler, features, model_name


def prepare_features(df, features, position):
    """
    Prepare feature matrix from raw data.
    
    Args:
        df: DataFrame with player data
        features: List of required features
        position: Position to filter for
    
    Returns:
        X: Feature matrix
        player_info: DataFrame with player metadata
    """
    # Filter position
    df_pos = df[df['position'] == position].copy()
    
    if len(df_pos) == 0:
        raise ValueError(f"No data found for position: {position}")
    
    # Extract metadata
    player_info = df_pos[['full_name', 'team', 'opponent', 'date']].copy()
    
    # Extract features
    missing_features = [f for f in features if f not in df_pos.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    X = df_pos[features].copy()
    
    # Handle missing values (fill with median or 0)
    X = X.fillna(X.median()).fillna(0)
    
    return X, player_info


def predict_with_confidence(model, scaler, X, model_name):
    """
    Make predictions with confidence scores.
    
    Args:
        model: Trained model
        scaler: Feature scaler (or None)
        X: Feature matrix
        model_name: Name of model type
    
    Returns:
        predictions, confidence_scores
    """
    # Scale if needed
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X_use = X_scaled
    else:
        X_use = X
    
    # Make predictions
    predictions = model.predict(X_use)
    
    # Calculate confidence
    # For tree models, use prediction variance across trees
    if hasattr(model, 'estimators_'):
        # RandomForest or GradientBoosting with multiple estimators
        try:
            tree_predictions = np.array([tree.predict(X_use) for tree in model.estimators_])
            prediction_std = np.std(tree_predictions, axis=0)
            # Convert std to confidence (0-1 scale)
            confidence = 1 / (1 + prediction_std)
        except:
            # Fallback if extracting tree predictions fails
            confidence = np.ones(len(predictions)) * 0.7  # Default moderate confidence
    else:
        # For linear models, use fixed confidence
        confidence = np.ones(len(predictions)) * 0.75
    
    return predictions, confidence


def make_predictions(data_path, position, horizons=['gw1', 'gw2', 'gw3']):
    """
    Make predictions for a position across multiple horizons.
    
    Args:
        data_path: Path to CSV with current player data
        position: Position to predict for
        horizons: List of horizons to predict
    
    Returns:
        DataFrame with predictions and confidence scores
    """
    print(f"\n{'='*80}")
    print(f"PREDICTIONS FOR {position.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✓ Loaded data: {len(df):,} rows")
    
    results = []
    
    for horizon in horizons:
        print(f"\n--- {horizon.upper()} Predictions ---")
        
        # Load model artifacts
        model, scaler, features, model_name = load_model_artifacts(position, horizon)
        
        # Prepare features
        X, player_info = prepare_features(df, features, position)
        print(f"  ✓ Prepared {len(X)} players")
        
        # Make predictions
        predictions, confidence = predict_with_confidence(model, scaler, X, model_name)
        
        # Store results
        horizon_results = player_info.copy()
        horizon_results[f'pred_{horizon}'] = predictions
        horizon_results[f'conf_{horizon}'] = confidence
        
        if len(results) == 0:
            results = horizon_results
        else:
            results = results.merge(
                horizon_results[[f'pred_{horizon}', f'conf_{horizon}']],
                left_index=True,
                right_index=True
            )
    
    # Add summary stats
    results['total_pred_3gw'] = (
        results['pred_gw1'] + results['pred_gw2'] + results['pred_gw3']
    )
    results['avg_confidence'] = (
        results['conf_gw1'] + results['conf_gw2'] + results['conf_gw3']
    ) / 3
    
    # Sort by total predicted points
    results = results.sort_values('total_pred_3gw', ascending=False)
    
    print(f"\n✓ Predictions complete!")
    return results


def display_top_predictions(results, top_n=10):
    """
    Display top predicted players.
    
    Args:
        results: DataFrame with predictions
        top_n: Number of top players to display
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_n} PREDICTED PLAYERS")
    print(f"{'='*80}\n")
    
    display_cols = [
        'full_name', 'team', 'opponent',
        'pred_gw1', 'pred_gw2', 'pred_gw3', 
        'total_pred_3gw', 'avg_confidence'
    ]
    
    top_players = results.head(top_n)[display_cols].copy()
    
    # Round for display
    for col in ['pred_gw1', 'pred_gw2', 'pred_gw3', 'total_pred_3gw']:
        top_players[col] = top_players[col].round(1)
    top_players['avg_confidence'] = top_players['avg_confidence'].round(3)
    
    print(top_players.to_string(index=False))
    print()


def main():
    """
    Example usage of the prediction pipeline.
    """
    print("\n" + "="*80)
    print("FPL PREDICTION SYSTEM - INFERENCE")
    print("="*80)
    
    # Path to your current gameweek data
    # This should have the same structure as masterFinalDataset.csv
    data_path = '../data/masterFinalDataset.csv'  # Replace with current GW data
    
    # Make predictions for each position
    positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
    all_results = {}
    
    for position in positions:
        try:
            results = make_predictions(data_path, position)
            all_results[position] = results
            display_top_predictions(results, top_n=10)
        except Exception as e:
            print(f"\n⚠️  Error predicting {position}: {e}\n")
            continue
    
    # Combine all positions
    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined = combined.sort_values('total_pred_3gw', ascending=False)
        
        print("\n" + "="*80)
        print("TOP 20 PLAYERS ACROSS ALL POSITIONS")
        print("="*80 + "\n")
        
        display_top_predictions(combined, top_n=20)
        
        # Save to CSV
        output_path = '../data/predictions_current_gw.csv'
        combined.to_csv(output_path, index=False)
        print(f"✓ All predictions saved to: {output_path}\n")
    
    print("="*80)
    print("PREDICTION COMPLETE!")
    print("="*80 + "\n")


def predict_single_player(player_name, data_path):
    """
    Make predictions for a single player.
    
    Args:
        player_name: Full name of player
        data_path: Path to data CSV
    
    Returns:
        DataFrame with player predictions
    """
    df = pd.read_csv(data_path)
    player_data = df[df['full_name'] == player_name]
    
    if len(player_data) == 0:
        print(f"Player not found: {player_name}")
        return None
    
    position = player_data['position'].iloc[0]
    print(f"\n{'='*80}")
    print(f"PREDICTIONS FOR: {player_name} ({position})")
    print(f"{'='*80}\n")
    
    results = make_predictions(data_path, position)
    player_results = results[results['full_name'] == player_name]
    
    if len(player_results) > 0:
        print("\nPlayer Predictions:")
        for horizon in ['gw1', 'gw2', 'gw3']:
            pred = player_results[f'pred_{horizon}'].iloc[0]
            conf = player_results[f'conf_{horizon}'].iloc[0]
            print(f"  {horizon.upper()}: {pred:.1f} points (confidence: {conf:.3f})")
        
        total = player_results['total_pred_3gw'].iloc[0]
        avg_conf = player_results['avg_confidence'].iloc[0]
        print(f"\n  Total (3 GW): {total:.1f} points")
        print(f"  Avg Confidence: {avg_conf:.3f}\n")
    
    return player_results


if __name__ == "__main__":
    # Run main prediction pipeline
    main()
    
    # Example: Predict for a specific player
    # Uncomment and modify player name:
    # predict_single_player('Mohamed Salah', '../data/masterFinalDataset.csv')
