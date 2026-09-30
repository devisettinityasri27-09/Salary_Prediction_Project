import mlflow
from mlflow.tracking import MlflowClient


def automate_champion_challenger():

    print(
        "[INFO] Starting Automated "
        "Model Lifecycle Manager..."
    )

    client = MlflowClient()

    model_name = (
        "Salary_Prediction_Production_Model"
    )

    metric_to_optimize = "r2_score"

    try:

        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        # 1. Move new models to Staging
        new_models = [
            mv
            for mv in all_versions
            if mv.current_stage == "None"
        ]

        if new_models:

            print(
                f"\n[INFO] Found {len(new_models)} "
                "new model(s). Moving them to Staging..."
            )

            for mv in new_models:

                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Staging",
                    archive_existing_versions=False
                )

                print(
                    f"  -> Version {mv.version} "
                    "is now in Staging."
                )

        # Refresh registry
        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        staging_models = [
            mv
            for mv in all_versions
            if mv.current_stage == "Staging"
        ]

        if not staging_models:

            print(
                "\n[INFO] No models in Staging "
                "to evaluate. Exiting."
            )

            return

        # 2. Find best challenger
        print(
            "\n[INFO] Evaluating Staging candidates..."
        )

        best_challenger = None
        best_challenger_score = float("-inf")

        for mv in staging_models:

            run = client.get_run(
                mv.run_id
            )

            score = run.data.metrics.get(
                metric_to_optimize,
                0.0
            )

            print(
                f"  -> Staging Candidate: "
                f"Version {mv.version} | "
                f"{metric_to_optimize}: {score:.4f}"
            )

            if score > best_challenger_score:

                best_challenger_score = score
                best_challenger = mv

        print(
            f"\n[INFO] Best Challenger is "
            f"Version {best_challenger.version} "
            f"({metric_to_optimize}: "
            f"{best_challenger_score:.4f})"
        )

        # 3. Find current Production model
        production_models = [
            mv
            for mv in all_versions
            if mv.current_stage == "Production"
        ]

        current_champion = (
            production_models[0]
            if production_models
            else None
        )

        promote_challenger = False

        if not current_champion:

            print(
                "[INFO] No model currently in Production. "
                "Challenger will be promoted."
            )

            promote_challenger = True

        else:

            champ_run = client.get_run(
                current_champion.run_id
            )

            champ_score = champ_run.data.metrics.get(
                metric_to_optimize,
                0.0
            )

            print(
                f"[INFO] Current Production Champion: "
                f"Version {current_champion.version} | "
                f"{metric_to_optimize}: {champ_score:.4f}"
            )

            # 4. Compare challenger and champion
            if best_challenger_score > champ_score:

                print(
                    "[SUCCESS] Challenger has a higher R²."
                )

                promote_challenger = True

            else:

                print(
                    "[INFO] Challenger did not exceed "
                    "the current Champion."
                )

        # 5. Promote challenger
        if promote_challenger:

            print(
                f"\n[INFO] Promoting Version "
                f"{best_challenger.version} to Production..."
            )

            client.transition_model_version_stage(
                name=model_name,
                version=best_challenger.version,
                stage="Production",
                archive_existing_versions=False
            )

            if current_champion:

                print(
                    f"[INFO] Archiving previous Champion "
                    f"(Version {current_champion.version})..."
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=current_champion.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

        # 6. Archive remaining staging models
        final_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        remaining_staging = [
            mv
            for mv in final_versions
            if mv.current_stage == "Staging"
        ]

        if remaining_staging:

            print(
                "\n[INFO] Cleaning up Staging area..."
            )

            for mv in remaining_staging:

                print(
                    f"  -> Archiving Version {mv.version}"
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

        print(
            "\n[SUCCESS] Automated Model Lifecycle "
            "execution complete!"
        )

    except Exception as e:

        print(
            f"[ERROR] Failed to automate lifecycle: {e}"
        )


if __name__ == "__main__":
    automate_champion_challenger()