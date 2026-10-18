# Yasal ve Etik Uyarı / Legal and Ethical Disclaimer

## Türkçe

Bu yazılım, bir lisans bitirme projesi kapsamında **akademik araştırma amacıyla**
geliştirilmiştir.

- **Tıbbi cihaz değildir.** Hiçbir ülkenin sağlık otoritesi tarafından onaylanmamıştır
  ve klinik kullanıma uygun değildir.
- **Teşhis aracı olarak kullanılamaz.** Hiçbir tıbbi karar bu yazılımın çıktısına
  dayandırılmamalıdır. Klinik değerlendirme yalnızca yetkili hekimler tarafından yapılır.
- **Deneysel çıktılar.** Model performansı, eğitimde kullanılan veri kümesine özgüdür;
  farklı hasta popülasyonlarında, farklı cihazlarda veya farklı çekim protokollerinde
  performans belirgin biçimde düşebilir. Modelin doğrulama verisi dışındaki verilerdeki
  davranışı ayrıca değerlendirilmemiştir.
- **Yanlış sonuç riski.** Sistem yanlış pozitif ve yanlış negatif üretebilir. Kaçırılan
  bir lezyon veya yanlış işaretlenen bir bölge söz konusu olabilir.

## Kişisel Veriler / Personal Data

Bu depo **hiçbir hasta verisi, DICOM dosyası, tıbbi rapor veya kişisel sağlık bilgisi
içermez.**

- Veri kümesi paylaşılmamıştır.
- Notebook çıktılarındaki örnek görüntüler, hastane PACS sisteminden **anonimleştirilerek**
  dışa aktarılmış kodlarla (ör. `Sub(93-164)0019`) etiketlenmiştir; hasta kimliğiyle
  ilişkilendirilemez.
- Model ağırlıkları, ayrıntılı hasta verisi içermez ancak eğitim verisinden kaynaklanan
  örüntüleri taşır; bu nedenle yayınlanmadan önce kurumsal etik kurul onayı ve veri
  sorumlusu görüşü alınmalıdır.

Sağlık verileri birçok ülkede özel nitelikli kişisel veri sayılır (Türkiye'de KVKK m.6,
AB'de GDPR m.9). Bu tür verilerle çalışırken ilgili mevzuata uyum sizin sorumluluğunuzdadır.

## English

This software was developed **for academic research purposes** as an undergraduate
graduation project.

- **Not a medical device.** It has not been cleared or approved by any regulatory
  authority and is not suitable for clinical use.
- **Not for diagnosis.** No medical decision should be based on its output. Clinical
  assessment is performed solely by qualified physicians.
- **Experimental results.** Model performance is specific to the training dataset and
  may degrade substantially across populations, scanners, and acquisition protocols.
- **Risk of error.** The system can produce false positives and false negatives.

This repository contains **no patient data, DICOM files, or personal health
information**. Sample images in notebook outputs were anonymized at the point of export
from the hospital PACS. As health data is classified as special-category personal data
under the GDPR (Art. 9) and similar regulations, compliance with applicable law remains
the responsibility of the user.

---

Bu yazılımı kullanarak bu koşulları kabul etmiş sayılırsınız.
By using this software you acknowledge and accept these terms.
