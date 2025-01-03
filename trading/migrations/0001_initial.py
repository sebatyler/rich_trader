# Generated by Django 5.1.4 on 2024-12-29 12:45

import django.db.models.deletion
import django.utils.timezone
import model_utils.fields
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='TradingConfig',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', model_utils.fields.AutoCreatedField(default=django.utils.timezone.now, editable=False, verbose_name='created')),
                ('modified', model_utils.fields.AutoLastModifiedField(default=django.utils.timezone.now, editable=False, verbose_name='modified')),
                ('coinone_access_key', models.CharField(max_length=255)),
                ('coinone_secret_key', models.CharField(max_length=255)),
                ('telegram_chat_id', models.CharField(max_length=255)),
                ('is_active', models.BooleanField(default=True)),
                ('target_coins', models.JSONField(help_text='List of target coins for each iteration')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Trading Configuration',
                'verbose_name_plural': 'Trading Configurations',
            },
        ),
    ]
