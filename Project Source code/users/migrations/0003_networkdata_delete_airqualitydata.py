# Generated by Django 5.0.4 on 2024-07-06 11:08

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0002_airqualitydata_delete_userpredictmodel"),
    ]

    operations = [
        migrations.CreateModel(
            name="NetworkData",
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
                ("is_CH", models.BooleanField(default=False)),
                ("who_CH", models.CharField(max_length=255)),
                ("dist_to_CH", models.FloatField()),
                ("adv_s", models.FloatField()),
                ("adv_r", models.FloatField()),
                ("join_s", models.FloatField()),
                ("join_r", models.FloatField()),
                ("sch_s", models.FloatField()),
                ("sch_r", models.FloatField()),
                ("rank", models.IntegerField()),
                ("data_s", models.FloatField()),
                ("data_r", models.FloatField()),
                ("data_sent_to_bs", models.FloatField()),
                ("dist_ch_to_bs", models.FloatField()),
                ("send_code", models.CharField(max_length=255)),
                ("expanded_energy", models.FloatField()),
                ("label", models.CharField(max_length=200)),
            ],
        ),
        migrations.DeleteModel(
            name="AirQualityData",
        ),
    ]
