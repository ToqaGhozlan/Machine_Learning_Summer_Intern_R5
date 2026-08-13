from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Prediction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("model_choice", models.CharField(max_length=20)),
                ("predicted_temp", models.FloatField()),
                ("lower", models.FloatField(help_text="Lower bound of the 95% confidence interval")),
                ("upper", models.FloatField(help_text="Upper bound of the 95% confidence interval")),
                ("from_cache", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
