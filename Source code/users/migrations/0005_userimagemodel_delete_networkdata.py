# Generated by Django 5.0.4 on 2024-07-12 05:13

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0004_alter_networkdata_is_ch_alter_networkdata_send_code_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="UserImageModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="predict/")),
                ("label", models.CharField(default="data", max_length=255)),
            ],
        ),
        migrations.DeleteModel(
            name="NetworkData",
        ),
    ]
